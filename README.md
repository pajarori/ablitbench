<div align="center">

# ablitbench

Security assessments for AI model files and API endpoints.

<pre>
  ▌ ▜ ▘▌
▀▌▛▌▐ ▌▛▌
█▌▙▌▐▖▌▙▌
</pre>

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Core](https://github.com/pajarori/ablitbench/actions/workflows/core.yml/badge.svg)](https://github.com/pajarori/ablitbench/actions/workflows/core.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

</div>

## Install

Requires Python 3.12+ and pipx.

```bash
pipx install git+https://github.com/pajarori/ablitbench.git
```

## Usage

Scan local files or a Hugging Face model:

```bash
ablib source ./models/suspect
ablib source ./models/model.gguf
ablib source hf://owner/model
```

Checks include loader code, metadata, unsafe serialization, archives, and Safetensors/GGUF weights. Static scans do not execute model code.

Assess an API endpoint:

```bash
ablib api https://provider.example/v1 --model model-name
```

Set credentials through `ABLIB_API_KEY` or `HF_TOKEN`. API scans send inference requests and may incur provider costs. OpenAI Chat/Responses and Anthropic Messages protocols are supported.

Compare weights, export JSON, or inspect available options:

```bash
ablib source ./models/suspect --reference ./models/base
ablib source ./models/suspect --format json > report.json
ablib api --help
ablib source --help
```

Use `--config FILE` for YAML/JSON configuration, `--plan` to preview a scan, and `--resume DIR` to continue a compatible run.

Source inference and real tool execution require Docker/Podman, a digest-pinned [worker image](workers/Dockerfile), and explicit memory/CPU/PID capacity. Build workers from the same core wheel as the CLI. Tools use a labeled simulator by default.

## Results

Runs are saved under `ablitbench-results/` with JSON/HTML reports, evidence, and coverage. Use `--output DIR` to choose another location.

Findings describe observed behavior and indicators; they do not prove poisoning or model substitution. Incomplete checks remain visible in coverage.

## License

[Apache-2.0](LICENSE). See [NOTICE](NOTICE) for third-party notices.
