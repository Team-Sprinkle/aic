# The same pinned simulator/interface base used by the reproduction runner.
FROM mambaorg/micromamba:2.3.3@sha256:800e7ade3ffe29c9a9ac2026163131495f8197c3852e572c5835beb4e8a33cd6 AS mamba
FROM ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923
USER root
COPY --from=mamba /bin/micromamba /usr/local/bin/micromamba
SHELL ["/bin/bash", "-c"]
COPY docker/final_controller/conda-linux-64.lock docker/final_controller/requirements.lock /opt/aic/dependencies/
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip && \
    micromamba create --yes --prefix /opt/aic/runtime --file /opt/aic/dependencies/conda-linux-64.lock && \
    python3 -m pip --python /opt/aic/runtime/bin/python install --no-cache-dir --no-deps -r /opt/aic/dependencies/requirements.lock && \
    /opt/aic/runtime/bin/python -m pip check && micromamba clean --all --yes
COPY aic_example_policies/aic_example_policies/final /opt/aic/source/aic_example_policies/final
RUN touch /opt/aic/source/aic_example_policies/__init__.py
COPY aic_model/aic_model /opt/aic/source/aic_model
# Supply an unpacked checksum-verified artifact bundle as a named build context.
COPY --from=artifacts / /opt/aic/artifacts/
COPY --chmod=755 docker/final_controller/entrypoint.sh /opt/aic/entrypoint.sh
ENV AIC_ARTIFACT_ROOT=/opt/aic/artifacts AIC_DEVICE=cuda:0 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ENTRYPOINT ["/opt/aic/entrypoint.sh"]
