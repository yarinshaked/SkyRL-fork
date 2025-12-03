FROM novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8

ENV SRC_DIR=/SkyRL-fork/skyrl-train
ENV HYDRA_FULL_ERROR=1
ENV HF_HOME=hf_cache/
ENV ENTRYPOINT_SCRIPT=./entrypoint.sh

USER root
WORKDIR /
RUN git clone https://github.com/yarinshaked/SkyRL-fork.git \
    && cd SkyRL-fork \
    && git checkout thesis

WORKDIR $SRC_DIR

ENTRYPOINT ["/bin/bash", "-c", "$ENTRYPOINT_SCRIPT"]

CMD ["./run_spartan.sh"]