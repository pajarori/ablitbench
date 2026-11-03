ARG BASE_IMAGE=python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
FROM ${BASE_IMAGE} AS ml
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/matplotlib
COPY workers/parity-probes-ml.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.lock && python -m pip check
ARG CORE_WHEEL=dist/ablitbench-*.whl
COPY ${CORE_WHEEL} /wheel/
RUN python -c "from pathlib import Path; assert len(list(Path('/wheel').glob('ablitbench-*.whl'))) == 1" && python -m pip install --no-cache-dir --no-deps /wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]

FROM ${BASE_IMAGE} AS garak
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/tmp/hf MPLCONFIGDIR=/tmp/matplotlib
COPY workers/parity-probes-garak.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.lock && python -m pip check
ARG CORE_WHEEL=dist/ablitbench-*.whl
COPY ${CORE_WHEEL} /wheel/
RUN python -c "from pathlib import Path; assert len(list(Path('/wheel').glob('ablitbench-*.whl'))) == 1" && python -m pip install --no-cache-dir --no-deps /wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]
