ARG RUST_IMAGE=rust@sha256:2775a09d208ff0d7c1f50490c45b62db929e87ba1dcbc3f2132ac71a704bcdd3
ARG BASE_IMAGE=python@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
FROM ${RUST_IMAGE} AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ pkg-config patch && rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:25a4f35b5d5331798bf03666738a7592ffb9530ef6d5db83a5d32dc977a33a97 https://codeload.github.com/izm1chael/layerfault/tar.gz/5701e15951773184e5a32e1553ac13c687ac67f8 /tmp/layerfault.tar.gz
RUN mkdir /build && tar -xzf /tmp/layerfault.tar.gz --strip-components=1 -C /build
WORKDIR /build
COPY workers/parity-source-layerfault.patch /tmp/layerfault.patch
RUN echo "51b0fcad1454e1f81715148dda0288e26964617094e09f0e97094b05d6f72f68  /tmp/layerfault.patch" | sha256sum -c - && patch -p1 < /tmp/layerfault.patch
COPY workers/parity-source-layerfault-hub.patch /tmp/layerfault-hub.patch
RUN echo "4a7f11d419e7902800cb542ce6d70a2509239113dc74c3c115cb7c9c9d1782d3  /tmp/layerfault-hub.patch" | sha256sum -c - && patch -p1 < /tmp/layerfault-hub.patch
COPY workers/parity-source-layerfault-judge.rs /tmp/judge-probe.rs
ARG BUILD_JOBS=1
RUN --mount=type=cache,target=/usr/local/cargo/registry --mount=type=cache,target=/usr/local/cargo/git --mount=type=cache,id=layerfault-5701e159-target,target=/build/target CARGO_BUILD_JOBS=${BUILD_JOBS} cargo build --locked --release --bin layerfault --lib && install -Dm755 target/release/layerfault /out/layerfault && cp src/judge.rs /out/judge.rs && cp src/platform/weekly.rs /out/weekly.rs && cp src/hub/client.rs /out/hub-client.rs && cp src/hub/client/range.rs /out/hub-range.rs && rustc --version > /out/rustc-version && rustc /tmp/judge-probe.rs --crate-name layerfault_judge_probe --edition=2021 --extern layerfault=target/release/liblayerfault.rlib -L dependency=target/release/deps -o /out/layerfault-judge-probe

FROM ${BASE_IMAGE}
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf DO_NOT_TRACK=1
COPY workers/core.lock /tmp/core.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/core.lock && apt-get update && apt-get install -y --no-install-recommends bubblewrap strace procps util-linux && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/layerfault /opt/parity/bin/layerfault
COPY --from=build /build/LICENSE /opt/parity/layerfault-LICENSE
COPY --from=build /out/judge.rs /opt/parity/source/judge.rs
COPY --from=build /out/weekly.rs /opt/parity/source/weekly.rs
COPY --from=build /out/hub-client.rs /opt/parity/source/hub-client.rs
COPY --from=build /out/hub-range.rs /opt/parity/source/hub-range.rs
COPY --from=build /out/rustc-version /opt/parity/source/rustc-version
RUN python -c "import hashlib,json; from pathlib import Path; p=Path('/opt/parity/bin/layerfault'); Path('/opt/parity/layerfault-build.json').write_text(json.dumps({'source_revision':'5701e15951773184e5a32e1553ac13c687ac67f8','source_archive_sha256':'25a4f35b5d5331798bf03666738a7592ffb9530ef6d5db83a5d32dc977a33a97','binary_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'broker_transport_patch':'51b0fcad1454e1f81715148dda0288e26964617094e09f0e97094b05d6f72f68','patched_source_sha256':hashlib.sha256(Path('/opt/parity/source/judge.rs').read_bytes()).hexdigest(),'smtp_source_sha256':hashlib.sha256(Path('/opt/parity/source/weekly.rs').read_bytes()).hexdigest(),'hub_transport_patch':'4a7f11d419e7902800cb542ce6d70a2509239113dc74c3c115cb7c9c9d1782d3','hub_source_sha256':hashlib.sha256(Path('/opt/parity/source/hub-client.rs').read_bytes()).hexdigest(),'hub_range_source_sha256':hashlib.sha256(Path('/opt/parity/source/hub-range.rs').read_bytes()).hexdigest(),'compiler':Path('/opt/parity/source/rustc-version').read_text().strip()}))"
ARG CORE_WHEEL=dist/ablitbench-*.whl
COPY ${CORE_WHEEL} /tmp/wheel/
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheel/ablitbench-*.whl
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]
