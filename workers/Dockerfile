ARG BASE_IMAGE=python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
FROM ${BASE_IMAGE} AS wheel
ARG CORE_WHEEL=dist/ablitbench-*.whl
COPY ${CORE_WHEEL} /wheel/
RUN python -c "from pathlib import Path; assert len(list(Path('/wheel').glob('ablitbench-*.whl'))) == 1, 'Select one wheel with --build-arg CORE_WHEEL=dist/<wheel>'"

FROM ${BASE_IMAGE} AS dependencies
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf
COPY workers/core.lock /tmp/core.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/core.lock && mkdir /fixture && printf 'The project color is blue.\n' > /fixture/public.txt

FROM dependencies AS tool-dependencies
RUN apt-get update && apt-get install -y --no-install-recommends strace && rm -rf /var/lib/apt/lists/*

FROM tool-dependencies AS core
COPY --from=wheel /wheel/ /tmp/wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]

FROM dependencies AS inference-dependencies
COPY workers/inference.lock /tmp/inference.lock
RUN python -m pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/inference.lock

FROM inference-dependencies AS inference
COPY --from=wheel /wheel/ /tmp/wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]

FROM rust@sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36 AS research-rust

FROM ${BASE_IMAGE} AS research-build
COPY --from=research-rust /usr/local/cargo /usr/local/cargo
COPY --from=research-rust /usr/local/rustup /usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo RUSTUP_HOME=/usr/local/rustup PATH=/usr/local/cargo/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends build-essential pkg-config && rm -rf /var/lib/apt/lists/*
COPY workers/research-build.lock /tmp/research-build.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/research-build.lock
COPY workers/research.lock /tmp/research.lock
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/usr/local/cargo/registry python -m pip wheel --no-build-isolation --config-settings=build-args=--locked --require-hashes -r /tmp/research.lock --wheel-dir /research-wheels

FROM ${BASE_IMAGE} AS research-dependencies
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PROMPTFOO_DISABLE_TELEMETRY=1 NO_ANALYTICS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1 HF_HOME=/tmp/hf
COPY --from=research-build /research-wheels /tmp/research-wheels
RUN python -m pip install --no-cache-dir --no-index --no-deps /tmp/research-wheels/*.whl && python -m pip check && rm -rf /tmp/research-wheels

FROM research-dependencies AS research
COPY --from=wheel /wheel/ /tmp/wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]

FROM ${BASE_IMAGE} AS gguf-dependencies
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_HOME=/tmp/hf
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake ninja-build && rm -rf /var/lib/apt/lists/*
COPY workers/gguf.lock /tmp/gguf.lock
ARG BUILD_JOBS=2
RUN CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_OPENMP=OFF" python -m pip install -v --no-cache-dir --require-hashes -r /tmp/gguf.lock

FROM gguf-dependencies AS gguf
COPY --from=wheel /wheel/ /tmp/wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]

FROM ${BASE_IMAGE} AS discovery-source
COPY workers/research-build.lock /tmp/build.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/build.lock
COPY workers/discovery-engine.lock /tmp/discovery-engine.lock
RUN --mount=type=cache,target=/root/.cache/pip python -m pip wheel --no-deps --no-build-isolation --require-hashes -r /tmp/discovery-engine.lock --wheel-dir /discovery-wheel

FROM inference-dependencies AS discovery-dependencies
COPY workers/discovery.lock /tmp/discovery.lock
RUN python -m pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/discovery.lock && python -m pip check
COPY --from=discovery-source /discovery-wheel /tmp/discovery-wheel
RUN python -m pip install --no-cache-dir --no-deps /tmp/discovery-wheel/bdrscan-*.whl
ENV ABLIB_DISCOVERY_SCOPE=cpu-exp1-no-cloud-no-codeql MPLCONFIGDIR=/tmp/matplotlib

FROM discovery-dependencies AS discovery
COPY --from=wheel /wheel/ /tmp/wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]

FROM inference-dependencies AS activation-dependencies
COPY workers/activation.lock /tmp/activation.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/activation.lock
ADD --checksum=sha256:a629872e19731bc448a06f747fb3a0bc14c417ac7c3baccb0c09522186d7d9be https://codeload.github.com/andyrdt/refusal_direction/tar.gz/9d852fae1a9121c78b29142de733cb1340770cc3 /tmp/refusal-direction.tar.gz
RUN mkdir -p /opt/refusal_direction && tar -xzf /tmp/refusal-direction.tar.gz --strip-components=1 -C /opt/refusal_direction && rm /tmp/refusal-direction.tar.gz

FROM activation-dependencies AS activation
COPY --from=wheel /wheel/ /tmp/wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]

FROM inference-dependencies AS bait
COPY --from=wheel /wheel/ /tmp/wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]

FROM inference-dependencies AS fingerprint-dependencies
ADD --checksum=sha256:a0810081d2363f3a599cfa7611d9239f3dd09de498a94281a5047aaca031b5f8 https://codeload.github.com/pasquini-dario/LLMmap/tar.gz/f661d550e3409d1bdf7e00b34bcec15296c5215e /tmp/llmmap.tar.gz
RUN mkdir -p /opt/llmmap && tar -xzf /tmp/llmmap.tar.gz --strip-components=1 -C /opt/llmmap && rm /tmp/llmmap.tar.gz && rm -rf /opt/llmmap/data

FROM fingerprint-dependencies AS fingerprint
COPY --from=wheel /wheel/ /tmp/wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]
