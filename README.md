<div align="center">

# ablitbench

Evidence-based assessments for AI model artifacts and API endpoints.

<pre>
  ▌ ▜ ▘▌
▀▌▛▌▐ ▌▛▌
█▌▙▌▐▖▌▙▌
</pre>

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Core](https://github.com/pajarori/ablitbench/actions/workflows/core.yml/badge.svg)](https://github.com/pajarori/ablitbench/actions/workflows/core.yml)
[![Stars](https://img.shields.io/github/stars/pajarori/ablitbench?style=flat&logo=github)](https://github.com/pajarori/ablitbench/stargazers)
[![Issues](https://img.shields.io/github/issues/pajarori/ablitbench?style=flat&logo=github)](https://github.com/pajarori/ablitbench/issues)
[![Last Commit](https://img.shields.io/github/last-commit/pajarori/ablitbench?style=flat&logo=github)](https://github.com/pajarori/ablitbench/commits/main)

</div>

## Installation

```bash
pipx install git+https://github.com/pajarori/ablitbench.git
```

To upgrade an existing installation:

```bash
pipx upgrade ablitbench
```

Python 3.12+ is required. The package is `ablitbench`; the command is `ablib`.

## Usage

Scan a local model directory or file without running model code:

```bash
ablib source ./models/suspect
ablib source ./models/model.gguf
```

Inspect a Hugging Face repository at a selected revision:

```bash
ablib source hf://owner/model --revision COMMIT --plan
```

Assess an API endpoint:

```bash
ablib api https://provider.example/v1 --model model-name
```

API scans send inference requests and may incur provider costs. Authentication uses `ABLIB_API_KEY`, or the environment variable selected with `--key-env`. Hugging Face uses `HF_TOKEN`. Credentials are not accepted as literal flags.

Compare compatible weights or export a report:

```bash
ablib source ./models/suspect --reference ./models/base
ablib source ./models/suspect --format json > report.json
ablib api --config providers.yaml --format jsonl > events.jsonl
```

Resume a compatible run or replay saved cases as a new run:

```bash
ablib source ./models/suspect --resume ./ablitbench-results/RUN
ablib api --config providers.yaml --replay ./ablitbench-results/RUN
```

### Options

| Flag | Description |
|------|-------------|
| `--config FILE` | YAML/JSON configuration, including multiple targets |
| `--checks IDS` | Comma-separated check IDs or categories |
| `--output DIR` | Run directory; default is a new directory under `ablitbench-results/` |
| `--format` | `terminal`, `json`, or `jsonl` |
| `--plan` | Resolve targets and show the plan without inference |
| `--resume DIR` | Continue a run with matching config, source, and implementation identities |
| `--replay DIR` | Run saved cases again with the current invocation's permissions |
| `--seed N` | Generation seed where the backend supports it |
| `--execute-tools` | Execute supported tools in a configured sandbox |
| `--model ID` | API model name |
| `--key-env NAME` | API credential environment variable; default `ABLIB_API_KEY` |
| `--protocol` | `auto`, `openai-chat`, `openai-responses`, or `anthropic-messages` |
| `--revision REV` | Hugging Face revision, resolved to an immutable commit |
| `--reference TARGET` | Source weights comparison target |
| `--offline` | Use local files and the Hugging Face cache |
| `--inference` | Enable source inference in a configured sandbox |
| `--help` | Show command help and available check IDs |

A positional target and config `targets` cannot be combined. Scalar flags override configuration values. Paths in configuration are relative to the working directory.

## Configuration

Save a configuration as `providers.yaml`:

```yaml
targets:
  - location: https://provider.example/v1
    model: model-name
    protocol: openai-chat
checks: [endpoint, behavior]
max_output_tokens: 256
request_timeout: 90
```

```bash
ablib api --config providers.yaml
```

For source targets, use local paths or `hf://owner/model` as `location` and run `ablib source --config models.yaml`.

| Check | Purpose |
|-------|---------|
| `source` | Loader AST, metadata, dependencies, archive paths, pickle opcodes, shard references |
| `weights` | Safetensors/GGUF structure, full statistics, compatible reference differences |
| `endpoint` | JSON/SSE compatibility, stream termination, reported model identity |
| `behavior` | Benign/adversarial cases, per-run canary disclosure, native tool integrity |
| `discovery` | Paired candidate/control probes for conditional canary disclosure |
| `engines` | ModelAudit/ModelScan assessments through separate sandbox images |

Use `dataset: cases.jsonl` and `repeats` for repeated observations. Each dataset line follows the case schema: `id`, `prompt`, optional `expected`, `kind`, and `tools`. Discovery requires an explicit `discovery.candidates` and `discovery.controls` corpus.

Optional `evaluation.ground_truth` JSONL labels contain `target`, `check`, and boolean `positive`. Labels are evaluated after detection and must stay outside source target directories. Without labeled ground truth, runs do not calculate accuracy or recall.

`max_output_tokens` and `request_timeout` are explicit configuration values. The API client has no default request timeout. Anthropic and source inference require an output-token budget supplied by the user. A configured judge uses its own `judge.key_env`, defaulting to `ABLIB_JUDGE_KEY`; judge failures remain inconclusive.

### Source Inference and Tools

Source inference and real tool execution require Docker/Podman, digest-pinned worker images, and explicit memory/CPU/PID capacity. pipx isolates CLI dependencies; model execution uses the worker sandbox.

```bash
ablib source ./models/suspect --inference --config inference.yaml
ablib api --config providers.yaml --execute-tools
```

Save the worker configuration as `inference.yaml`, replacing the image digest:

```yaml
max_output_tokens: 256
runtime:
  engine: docker
  backend: transformers
  inference_image: sha256:REPLACE_WITH_LOCAL_IMAGE_ID
  memory_bytes: 2147483648
  cpus: 2
  pids: 128
```

`--execute-tools` also requires `runtime.tool_image`; source tools require `--inference`. The default tool loop uses a deterministic simulator and labels its actions `simulated`.

Worker backends include Transformers/PEFT and llama.cpp. For a PEFT adapter, set the target's `base` to an explicit local or Hugging Face base-model source. Native source tool parsing requires a compatible tokenizer response template. Backend and model support depend on the worker image. Worker images are built from the same core wheel. Select the image for the backend you intend to run.

### Worker Images

Clone the source, build its wheel, and use that same wheel for the CLI and worker:

```bash
git clone https://github.com/pajarori/ablitbench.git
cd ablitbench
uv build
pipx install --force dist/ablitbench-*.whl
docker build --target inference -f workers/Dockerfile -t ablitbench-inference:dev .
docker image inspect ablitbench-inference:dev --format '{{.Id}}'
```

Keep one core wheel in `dist/`, or select it explicitly with `--build-arg CORE_WHEEL=dist/<wheel>`. Use the returned `sha256:...` image ID in `runtime.inference_image`. Registry images must use `repository@sha256:...`.

| Build target | Runtime purpose |
|--------------|-----------------|
| `core` | Tool executor; configure `runtime.tool_image` |
| `inference` | Transformers/PEFT; configure `runtime.inference_image` |
| `gguf` | llama.cpp; use `runtime.backend: llama-cpp` |
| `research` | ModelAudit/ModelScan; configure `runtime.research_image` and `runtime.research_engines` |

The `workers/` directory contains optional image build recipes and dependency locks. pipx, API scans, and static source scans do not require it. Sandbox execution uses a non-root user, read-only mounts, no network, and explicit resource capacity. Worker and supervisor core identities must match; rebuild images after changing core code.

## Output Notes

- Each run stores `run.sqlite`, inventory, content-addressed evidence, `report.json`, standalone `report.html`, and `events.jsonl`.
- `severity`, evidence `strength`, attribution, and coverage are separate fields. There is no model honesty score.
- Coverage distinguishes `completed`, `not_requested`, `not_applicable`, `blocked`, `unsupported`, `error`, and `cancelled`.
- Source scans without inference do not test behavior. Offline cache inventory cannot establish repository completeness.
- Weight anomalies are indicators. Model-name differences and latency alone do not prove model substitution.
- Interrupted external requests are not retried blindly. Resume preserves ambiguous delivery as a gap.
- Evidence may contain model output or sensitive input. Run directories are private; reports do not include authentication headers.
- Exit codes: `0` completed without medium-or-higher findings, `1` findings, `2` invocation/fatal failure, `3` incomplete, `130` cancelled.

## Python API

```python
import asyncio

from ablitbench import ScanConfig, Scanner

async def main():
    config = ScanConfig.model_validate({
        "mode": "source",
        "targets": [{"location": "./models/suspect"}],
    })
    result = await Scanner().run(config)
    for finding in result.findings():
        print(finding.title)

asyncio.run(main())
```

`Scanner.plan(config)` performs inventory without inference. `Scanner.run(config, on_event=callback)` accepts a synchronous event callback. Findings can be read incrementally from `RunResult`.

## Development

```bash
uv sync --locked --dev
uv run ruff check ablitbench
uv run ruff format --check ablitbench
uv run mypy ablitbench
uv build
```

The Python package lives directly in `ablitbench/`. Local documentation, examples, tests, and research clones stay ignored and are excluded from distributions. GitHub Actions builds and smoke-tests the public package.

## Credits & References

- [subdotko](https://github.com/pajarori/subdotko) and [raider](https://github.com/pajarori/raider) — project layout and CLI presentation.
- [ModelAudit](https://github.com/promptfoo/modelaudit) and [ModelScan](https://github.com/protectai/modelscan) — optional artifact-scanning engines.

## License

[Apache License 2.0](LICENSE). Third-party models, datasets, and dependencies retain their own licenses; see [NOTICE](NOTICE).
