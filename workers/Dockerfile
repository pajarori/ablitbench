ARG BASE_IMAGE=python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
FROM ${BASE_IMAGE} AS wheel
ARG CORE_WHEEL=dist/ablitbench-*.whl
COPY ${CORE_WHEEL} /wheel/
RUN python -c "from pathlib import Path; assert len(list(Path('/wheel').glob('ablitbench-*.whl'))) == 1, 'Select one wheel with --build-arg CORE_WHEEL=dist/<wheel>'"

FROM ${BASE_IMAGE} AS dependencies
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf
COPY workers/core.lock /tmp/core.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/core.lock && mkdir /fixture && printf 'The project color is blue.\n' > /fixture/public.txt

FROM dependencies AS core
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

FROM ${BASE_IMAGE} AS research-dependencies
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PROMPTFOO_DISABLE_TELEMETRY=1 NO_ANALYTICS=1 DO_NOT_TRACK=1 HF_HUB_OFFLINE=1 HF_HOME=/tmp/hf
COPY workers/research.lock /tmp/research.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/research.lock

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
