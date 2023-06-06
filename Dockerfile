ARG CUDA_IMAGE=11.7.0-devel-ubuntu18.04
FROM nvidia/cuda:${CUDA_IMAGE} as cuda_source
FROM continuumio/miniconda3:22.11.1

# Proxy (if needed)
ARG HTTP_PROXY
ARG HTTPS_PROXY=${HTTP_PROXY}
ARG http_proxy=${HTTP_PROXY}
ARG https_proxy=${HTTP_PROXY}

# Build packages
RUN apt-get update \
    && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        ibverbs-providers \
        libibverbs1 \
        net-tools \
        librdmacm1 \
        libsndfile1 \
        openssh-client \
        openssh-server \
        software-properties-common \
        vim \
        g++ \
        unzip \
        file \
        man \
        file \
        neofetch \
        cmake \
    && apt-get clean

# Copy cuda devel from nvidia docker
ARG CUDA=11.7
ENV CUDA_HOME=/usr/local/cuda-${CUDA}
COPY --from=cuda_source ${CUDA_HOME} ${CUDA_HOME}
ENV PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}

COPY requirements/* /requirements/
WORKDIR /requirements

RUN conda config --set channel_priority strict && \
    conda env create -f conda-environment.yaml && conda clean -q -y -a && cd .. && rm -r /requirements

# Replace the shell binary with activated conda env
RUN echo "conda activate "`conda env list | tail -2 | cut -d' ' -f1` >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Default entrypoint
WORKDIR /
CMD ["/bin/bash", "--login"]
