ARG RUST_IMAGE=rust@sha256:2775a09d208ff0d7c1f50490c45b62db929e87ba1dcbc3f2132ac71a704bcdd3
ARG BASE_IMAGE=python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
FROM ${RUST_IMAGE} AS native-build-toolchain
FROM ${BASE_IMAGE} AS dependencies
ARG RUST_IMAGE
USER root
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf DO_NOT_TRACK=1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
ARG SOURCE_LOCK=workers/parity-source-runtime-cpu.lock
COPY ${SOURCE_LOCK} /tmp/runtime.lock
COPY workers/parity-source-build.lock /tmp/source-build.lock
COPY workers/parity-source-build-manifest.py /tmp/source-build-manifest.py
RUN --mount=from=native-build-toolchain,source=/usr/local/rustup,target=/opt/ablib-rustup,ro --mount=from=native-build-toolchain,source=/usr/local/cargo/bin,target=/opt/ablib-cargo-bin,ro python -m pip uninstall -y bdrscan datasets && apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev libgomp1 libstdc++6 && python -m pip install --no-cache-dir --require-hashes -r /tmp/source-build.lock && PATH=/opt/ablib-cargo-bin:$PATH RUSTUP_HOME=/opt/ablib-rustup RUSTUP_TOOLCHAIN=1.97.1 CARGO_HOME=/tmp/ablib-cargo python -m pip install --no-cache-dir --no-build-isolation --require-hashes -r /tmp/runtime.lock && PATH=/opt/ablib-cargo-bin:$PATH RUSTUP_HOME=/opt/ablib-rustup RUSTUP_TOOLCHAIN=1.97.1 CARGO_HOME=/tmp/ablib-cargo python /tmp/source-build-manifest.py --runtime-lock /tmp/runtime.lock --build-lock /tmp/source-build.lock --toolchain-image "$RUST_IMAGE" --output /opt/parity/source-build.json && python -m pip uninstall -y hatchling maturin trove-classifiers && apt-get purge -y gcc libc6-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* /root/.cache/puccinialin /root/.cache/pip /tmp/ablib-cargo
FROM dependencies
ARG CORE_WHEEL=dist/ablitbench-*.whl
COPY ${CORE_WHEEL} /tmp/wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl && python -m pip check
COPY workers/parity-source-format-validation.py /opt/parity/format-validation.py
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]
