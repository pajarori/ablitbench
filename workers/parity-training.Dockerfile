ARG BASE_IMAGE=python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY workers/core.lock workers/discovery.lock workers/parity-training-web.lock /tmp/locks/
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/locks/core.lock && python -m pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/locks/discovery.lock && python -m pip install --no-cache-dir --require-hashes -r /tmp/locks/parity-training-web.lock
ARG CORE_WHEEL=dist/ablitbench-*.whl
COPY ${CORE_WHEEL} /tmp/wheel/
RUN python -c "from pathlib import Path; assert len(list(Path('/tmp/wheel').glob('ablitbench-*.whl'))) == 1" && python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]
