ARG NATIVE_IMAGE
FROM ${NATIVE_IMAGE}
USER root
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-17 && rm -rf /var/lib/apt/lists/*
USER 65534:65534
CMD ["python", "-m", "ablitbench.worker"]
