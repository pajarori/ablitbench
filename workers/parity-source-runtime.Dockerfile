ARG BASE_IMAGE=python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf DO_NOT_TRACK=1
ARG SOURCE_LOCK=workers/parity-source-runtime.lock
COPY ${SOURCE_LOCK} /tmp/runtime.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/runtime.lock
ARG CORE_WHEEL=dist/ablitbench-*.whl
COPY ${CORE_WHEEL} /tmp/wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]
