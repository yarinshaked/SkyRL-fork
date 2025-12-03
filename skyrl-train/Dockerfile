FROM novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8

ENV SRC_DIR=/SkyRL-fork/skyrl-train
ENV HYDRA_FULL_ERROR=1
ENV HF_HOME=/hf_cache

USER root
WORKDIR /
RUN git clone https://github.com/yarinshaked/SkyRL-fork.git \
    && cd SkyRL-fork \
    && git checkout thesis

WORKDIR $SRC_DIR

RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]

CMD ["./run_spartan.sh"]