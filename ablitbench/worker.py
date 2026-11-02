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

from ablitbench.models import Generation, WorkerRequest, WorkerResponse
from ablitbench.provenance import implementation_identity


class Inference:
    def __init__(self, device: str = "cpu") -> None:
        transformers = importlib.import_module("transformers")
        self.torch = importlib.import_module("torch")
        self.torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
        self.device = device
        if device == "cuda" and not self.torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; CPU fallback is disabled")
        adapter = Path("/model/adapter_config.json").is_file()
        model_path = "/base" if adapter else "/model"
        if adapter and not Path("/base/config.json").is_file():
            raise ValueError("Adapter inference requires an explicitly resolved base model")
        tokenizer_path = "/model" if Path("/model/tokenizer_config.json").is_file() else model_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True, trust_remote_code=False
        )
        response_template = getattr(self.tokenizer, "response_template", None)
        if isinstance(response_template, dict) and response_template.get("start_anchor") == "":
            raise ValueError("Empty native response start_anchor is not supported")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        if adapter:
            if not Path("/model/adapter_model.safetensors").is_file():
                raise ValueError("Only Safetensors adapters are permitted")
            peft = importlib.import_module("peft")
            self.model = peft.PeftModel.from_pretrained(
                self.model,
                "/model",
                local_files_only=True,
                is_trainable=False,
                use_safetensors=True,
            )
        self.model = self.model.to(device).eval()

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("tools") and not getattr(self.tokenizer, "response_template", None):
            raise ValueError("Tokenizer has no native response template for structured tool calls")
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
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
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
        if request.get("tools"):
            raw = self.tokenizer.decode(tokens, skip_special_tokens=False)
            result["raw_text"] = raw
            message = self.tokenizer.parse_response(
                raw, prefix=inputs["input_ids"][0], tools=request["tools"]
            )
            result["text"] = message.get("content") or ""
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
        return decode_json("openai-chat", result).model_dump()


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
    if request.name != "read_file" or set(request.arguments) != {"path"}:
        raise ValueError("Unsupported tool request")
    if request.arguments["path"] != "/fixture/public.txt":
        return {"error": "access denied", "level": "sandbox execution"}
    return {"content": Path("/fixture/public.txt").read_text(), "level": "sandbox execution"}


def main() -> None:
    model: Inference | GGUFInference | None = None
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
        print(response.model_dump_json(exclude_none=True), flush=True)


if __name__ == "__main__":
    main()
