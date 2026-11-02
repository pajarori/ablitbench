from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from ablitbench.models import TOOL_AUTHORIZATION_POLICY, Generation, WorkerRequest, WorkerResponse
from ablitbench.provenance import implementation_identity


class Inference:
    def __init__(
        self,
        device: str = "cpu",
        model_root: Path = Path("/model"),
        base_root: Path = Path("/base"),
    ) -> None:
        transformers = importlib.import_module("transformers")
        self.torch = importlib.import_module("torch")
        self.torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        self.device = device
        if device == "cuda" and not self.torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; CPU fallback is disabled")
        adapter = (model_root / "adapter_config.json").is_file()
        model_path = base_root if adapter else model_root
        if adapter and not (base_root / "config.json").is_file():
            raise ValueError("Adapter inference requires an explicitly resolved base model")
        tokenizer_path = (
            model_root if (model_root / "tokenizer_config.json").is_file() else model_path
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False
        )
        response_template = getattr(self.tokenizer, "response_template", None)
        fields = response_template.get("fields", {}) if isinstance(response_template, dict) else {}
        self.channel_fields = (
            set(fields)
            if isinstance(fields, dict)
            and {"content", "final"}.intersection(fields)
            and {"reasoning", "reasoning_content", "refusal"}.intersection(fields)
            else set()
        )
        self.native_tools = bool(
            isinstance(response_template, dict)
            and response_template
            and isinstance(fields, dict)
            and "tool_calls" in fields
            and response_template.get("start_anchor") != ""
            and callable(getattr(self.tokenizer, "parse_response", None))
        )
        if self.native_tools:
            parser_factory = getattr(self.tokenizer, "get_response_parser", None)
            if not callable(parser_factory):
                self.native_tools = False
            else:
                try:
                    parser_factory(prefix="")
                except (ValueError, TypeError):
                    self.native_tools = False
        self.model, loading = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            output_loading_info=True,
        )
        incomplete = {
            key: loading.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading.get(key)
        }
        if incomplete:
            raise ValueError(
                "Checkpoint parameter loading is incomplete: " + json.dumps(incomplete, default=str)
            )
        if adapter:
            adapter_path = model_root / "adapter_model.safetensors"
            if not adapter_path.is_file():
                raise ValueError("Only Safetensors adapters are permitted")
            peft = importlib.import_module("peft")
            self.model = peft.PeftModel.from_pretrained(
                self.model,
                model_root,
                local_files_only=True,
                is_trainable=False,
                use_safetensors=True,
            )
            state_function = getattr(peft, "get_peft_model_state_dict", None)
            if not callable(state_function):
                raise ValueError("PEFT cannot expose the normalized adapter state contract")
            expected = state_function(self.model)
            if not isinstance(expected, dict) or not expected:
                raise ValueError("PEFT adapter state contract is empty or unsupported")
            safetensors = importlib.import_module("safetensors")
            with safetensors.safe_open(adapter_path, framework="pt", device="cpu") as stored:
                keys = set(stored.keys())
                if keys != set(expected):
                    raise ValueError("Adapter parameter inventory is incomplete or unexpected")
                if any(
                    tuple(stored.get_slice(key).get_shape()) != tuple(value.shape)
                    for key, value in expected.items()
                ):
                    raise ValueError(
                        "Adapter parameter shapes differ from the normalized state contract"
                    )
        self.model = self.model.to(device).eval()

    def response_parser(self, input_ids: Any, tools: Any) -> Any:
        template = self.tokenizer.response_template
        fields = template.get("fields", {})
        if not isinstance(fields, dict):
            raise ValueError("Unsupported response template field shape")
        if self.channel_fields - {
            "content",
            "final",
            "reasoning",
            "reasoning_content",
            "refusal",
            "tool_calls",
        }:
            raise ValueError("Unsupported response template output fields")
        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        anchor = template.get("start_anchor")
        if template.get("start_anchor_pattern"):
            raise ValueError(
                "Regex response start anchors are unsupported for source normalization"
            )
        prefix = ""
        if anchor is not None:
            if not isinstance(anchor, str) or not anchor:
                raise ValueError("Unsupported response template start anchor")
            if anchor not in prompt:
                raise ValueError("Response template start anchor is absent from the prompt")
            prefix = prompt.rsplit(anchor, 1)[1]
        else:
            for field in fields.values():
                if not isinstance(field, dict):
                    raise ValueError("Unsupported response template field shape")
                opening = field.get("open")
                if isinstance(opening, str) and prompt.rstrip().endswith(opening):
                    prefix = opening
                    break
        remaining = prefix
        delimiters = sorted(
            {
                value
                for field in fields.values()
                if isinstance(field, dict)
                for key in ("open", "close")
                if isinstance(value := field.get(key), str) and value
            },
            key=len,
            reverse=True,
        )
        while remaining.strip():
            remaining = remaining.lstrip()
            delimiter = next((item for item in delimiters if remaining.startswith(item)), None)
            if delimiter is None:
                raise ValueError("Response template prefix contains ungenerated channel content")
            remaining = remaining[len(delimiter) :]
        parser = self.tokenizer.get_response_parser(prefix=prefix, tools=tools)
        if any(event.get("text", "").strip() for event in parser.initial_events):
            raise ValueError("Response template prefix contains ungenerated channel content")
        return parser

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("tools") and not self.native_tools:
            raise ValueError("Tokenizer does not support native structured tool calls")
        if request.get("seed") is not None:
            self.torch.manual_seed(request["seed"])
        inputs = self.tokenizer.apply_chat_template(
            request["messages"],
            tools=request.get("tools"),
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = inputs.to(self.device)
        encoded = self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        missing_markers = {
            "system-policy-not-encoded"
            if marker == TOOL_AUTHORIZATION_POLICY
            else "system-marker-not-encoded"
            for marker in request.get("required_system_markers", [])
            if marker not in encoded
        }
        parse_channels = bool(getattr(self, "channel_fields", set()))
        parser = (
            self.response_parser(inputs["input_ids"][0], request.get("tools"))
            if request.get("tools") or parse_channels
            else None
        )
        context_capacities = [
            value
            for value in (
                getattr(self.model.config, "max_position_embeddings", None),
                request.get("context_tokens"),
            )
            if type(value) is int and value > 0
        ]
        context_limit = min(context_capacities) if context_capacities else None
        if (
            context_limit
            and inputs["input_ids"].shape[-1] + request["max_output_tokens"] > context_limit
        ):
            raise ValueError(
                "Request exceeds model context capacity; no prompt truncation performed"
            )
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=request["max_output_tokens"], do_sample=False
            )
        tokens = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        result: dict[str, Any] = {
            "text": text,
            "finish": "length" if len(tokens) == request["max_output_tokens"] else "stop",
            "usage": {
                "input_tokens": int(inputs["input_ids"].shape[-1]),
                "output_tokens": len(tokens),
            },
        }
        if missing_markers:
            result["observations"] = [
                {
                    "code": code,
                    "detail": "A required system instruction was not preserved in encoded input",
                    "severity": "info",
                }
                for code in sorted(missing_markers)
            ]
        if parser is not None:
            raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
            result["raw_text"] = raw
            parser.feed(raw)
            message, _ = parser.finalize()
            if set(message) - {
                "role",
                "content",
                "final",
                "reasoning",
                "reasoning_content",
                "refusal",
                "tool_calls",
            }:
                raise ValueError("Unsupported parsed response output fields")
            for key in ("content", "final", "reasoning", "reasoning_content", "refusal"):
                if message.get(key) is not None and not isinstance(message[key], str):
                    raise ValueError("Unsupported parsed response channel shape: " + key)
            if message.get("content") is not None and message.get("final") is not None:
                raise ValueError("Response declares ambiguous final channels")
            if (
                message.get("reasoning") is not None
                and message.get("reasoning_content") is not None
            ):
                raise ValueError("Response declares ambiguous reasoning channels")
            result["text"] = message.get("content") or message.get("final") or ""
            result["reasoning"] = message.get("reasoning") or message.get("reasoning_content") or ""
            result["refusal"] = message.get("refusal") or ""
            result["tools"] = [
                {
                    "id": tool.get("id") or uuid.uuid4().hex,
                    "name": tool["function"]["name"],
                    "arguments": json.dumps(tool["function"]["arguments"])
                    if isinstance(tool["function"]["arguments"], dict)
                    else tool["function"]["arguments"],
                }
                for tool in message.get("tool_calls", [])
            ]
            if result["tools"] and result["finish"] != "length":
                result["finish"] = "tool_calls"
        return result


class GGUFInference:
    def __init__(self, request: WorkerRequest) -> None:
        llama = importlib.import_module("llama_cpp")
        if request.device == "cuda" and not llama.llama_supports_gpu_offload():
            raise ValueError("GGUF engine has no GPU offload support; CPU fallback disabled")
        models = list(Path("/model").glob("*.gguf"))
        if len(models) != 1:
            raise ValueError("GGUF inference requires a directory with exactly one selected GGUF")
        self.native_tools: bool | None = None
        self.model = llama.Llama(
            model_path=str(models[0]),
            n_ctx=request.context_tokens or 0,
            n_gpu_layers=-1 if request.device == "cuda" else 0,
            verbose=False,
            n_threads=int(os.environ.get("OMP_NUM_THREADS", "1")),
        )

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        from ablitbench.transport import decode_json

        result = self.model.create_chat_completion(
            messages=request["messages"],
            tools=request.get("tools"),
            max_tokens=request["max_output_tokens"],
            seed=request.get("seed"),
            stream=False,
        )
        generation = decode_json("openai-chat", result).model_dump()
        if request.get("required_system_markers"):
            codes = {
                "system-policy-encoding-unverified"
                if marker == TOOL_AUTHORIZATION_POLICY
                else "system-marker-encoding-unverified"
                for marker in request["required_system_markers"]
            }
            generation["observations"].extend(
                {
                    "code": code,
                    "detail": "Backend does not expose encoded input for marker verification",
                    "severity": "info",
                }
                for code in sorted(codes)
            )
        return generation


def isolation() -> None:
    if sys.platform != "linux":
        raise ValueError("Worker requires Linux and an unprivileged user")
    if os.getuid() == 0:
        raise ValueError("Worker requires Linux and an unprivileged user")
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    if (
        status["NoNewPrivs"].strip() != "1"
        or status["Seccomp"].strip() != "2"
        or int(status["CapEff"].strip(), 16)
    ):
        raise ValueError(
            "Worker requires no-new-privileges, seccomp filtering, and zero capabilities"
        )


def capacity(request: WorkerRequest) -> dict[str, Any]:
    root = Path("/sys/fs/cgroup")
    if (root / "memory.max").is_file():
        memory = (root / "memory.max").read_text().strip()
        pids = (root / "pids.max").read_text().strip()
        quota, period = (root / "cpu.max").read_text().split()
        cpus = float(quota) / float(period) if quota != "max" else None
    else:
        memory = (root / "memory/memory.limit_in_bytes").read_text().strip()
        pids = (root / "pids/pids.max").read_text().strip()
        quota = (root / "cpu/cpu.cfs_quota_us").read_text().strip()
        period = (root / "cpu/cpu.cfs_period_us").read_text().strip()
        cpus = float(quota) / float(period) if int(quota) > 0 else None
    if (
        not request.memory_bytes
        or memory == "max"
        or int(memory) > request.memory_bytes
        or not request.pids
        or pids == "max"
        or int(pids) > request.pids
        or not request.cpus
        or cpus is None
        or cpus > request.cpus + 0.000001
    ):
        raise ValueError("Runtime did not enforce requested memory, CPU, and PID isolation")
    return {"memory_bytes": int(memory), "pids": int(pids), "cpus": cpus}


def execute_tool(request: WorkerRequest) -> dict[str, Any]:
    from ablitbench.observer import execute

    return execute(request)


def main() -> None:
    model: Inference | GGUFInference | None = None
    activation_model: Any = None
    bait_model: Any = None
    fingerprint_model: Any = None
    for line in sys.stdin:
        response = WorkerResponse(
            isolation=platform.system().lower(), implementation=implementation_identity()
        )
        try:
            isolation()
            request = WorkerRequest.model_validate_json(line)
            response.capacity = capacity(request)
            if request.operation == "tool":
                response.result = execute_tool(request)
            elif request.operation == "scan":
                from ablitbench.engines import scan

                if request.engine is None or request.file is None:
                    raise ValueError("Engine and file are required")
                path = Path(request.file).resolve()
                if not path.is_relative_to("/model"):
                    raise ValueError("Scan input is outside model mount")
                with redirect_stdout(sys.stderr):
                    response.scan = scan(request.engine, path, request.timeout)
            elif request.operation == "discovery":
                from ablitbench.discovery_engine import run

                if request.settings.get("device", request.device) != request.device:
                    raise ValueError("Discovery device conflicts with runtime device")
                if (
                    request.context_tokens is not None
                    and request.settings.get("context_tokens", request.context_tokens)
                    != request.context_tokens
                ):
                    raise ValueError("Discovery context capacity conflicts with runtime")
                settings = request.settings | {"device": request.device}
                if request.context_tokens is not None:
                    settings["context_tokens"] = request.context_tokens
                if not Path("/output").is_dir():
                    raise ValueError("Discovery output mount is required")
                with redirect_stdout(sys.stderr):
                    response.discovery = run(Path("/model"), Path("/output"), settings)
            elif request.operation == "activation":
                from ablitbench.activation_engine import ActivationEngine

                if not Path("/output").is_dir():
                    raise ValueError("Activation output mount is required")
                with redirect_stdout(sys.stderr):
                    if request.phase == "fit":
                        if activation_model is not None:
                            raise ValueError("Activation engine already fitted; no implicit refit")
                        activation_model = ActivationEngine(
                            device=request.device, context_tokens=request.context_tokens
                        )
                        response.activation = activation_model.fit(request.settings)
                    else:
                        if (
                            activation_model is None
                            or request.prompt is None
                            or request.group is None
                        ):
                            raise ValueError("Fitted activation engine, prompt and group required")
                        response.activation = activation_model.evaluate(
                            request.prompt, request.group
                        )
            elif request.operation == "bait":
                from ablitbench.bait_engine import BaitEngine

                if not Path("/output").is_dir():
                    raise ValueError("Inversion output mount is required")
                with redirect_stdout(sys.stderr):
                    if request.phase == "configure":
                        if bait_model is not None:
                            raise ValueError(
                                "Inversion engine already configured; no implicit reset"
                            )
                        bait_model = BaitEngine(
                            device=request.device, context_tokens=request.context_tokens
                        )
                        response.bait = bait_model.configure(request.settings)
                    elif request.phase == "search" and bait_model is not None:
                        response.bait = bait_model.search(request.token_ids)
                    else:
                        raise ValueError("Configured inversion engine and search phase required")
            elif request.operation == "fingerprint":
                from ablitbench.fingerprint_engine import FingerprintEngine

                if not Path("/output").is_dir():
                    raise ValueError("Fingerprint output mount is required")
                with redirect_stdout(sys.stderr):
                    if request.phase == "configure":
                        if fingerprint_model is not None:
                            raise ValueError("Fingerprint engine already configured")
                        fingerprint_model = FingerprintEngine()
                        response.fingerprint = fingerprint_model.configure(request.settings)
                    elif (
                        request.phase == "evaluate"
                        and fingerprint_model is not None
                        and len(request.responses) == 1
                    ):
                        response.fingerprint = fingerprint_model.evaluate(request.responses[0])
                    elif request.phase == "finalize" and fingerprint_model is not None:
                        response.fingerprint = fingerprint_model.summary()
                    else:
                        raise ValueError("Configured fingerprint engine and valid phase required")
            else:
                if model is None:
                    model = (
                        GGUFInference(request)
                        if request.backend == "llama-cpp"
                        else Inference(request.device)
                    )
                response.generation = Generation.model_validate(
                    model.generate(request.model_dump())
                )
        except Exception as error:
            response.error = f"{type(error).__name__}: {error}"
        response.model_initialized = model is not None
        if model is not None:
            response.native_tools = model.native_tools
        print(response.model_dump_json(exclude_none=True), flush=True)


if __name__ == "__main__":
    main()
