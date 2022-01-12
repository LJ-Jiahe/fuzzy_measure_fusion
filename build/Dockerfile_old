FROM nvidia/cuda:11.4.0-base-ubuntu20.04

LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

ENV NCCL_VERSION 2.10.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-11-4=11.4.0-1 \
    libnpp-11-4=11.4.0.33-1 \
    cuda-nvtx-11-4=11.4.43-1 \
    libcublas-11-4=11.5.2.43-1 \
    libcusparse-11-4=11.6.0.43-1 \
    libnccl2=$NCCL_VERSION-1+cuda11.4 \
    && rm -rf /var/lib/apt/lists/*

# apt from auto upgrading the cublas package. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88

RUN apt-mark hold libcublas-11-4 libnccl2

#-----------------------------
# Initialize: Basic Packages
#-----------------------------

ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y tmux vim wget htop git cmake libgl1-mesa-glx

#-----------------------------
# Initialize: Conda Workflow
# - Python Virtual Environment
#-----------------------------

WORKDIR develop/build
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

RUN bash Miniconda3-latest-Linux-x86_64.sh -p /develop/build/miniconda3 -b
ENV PATH=/develop/build/miniconda3/bin:${PATH}
RUN conda update -y conda

RUN conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
RUN conda install -c conda-forge cvxopt qpsolvers matplotlib tqdm
RUN conda install -c anaconda numpy scipy

#-----------------------------
# Load: Working Git Project
# - Remember: Generate Token
#-----------------------------

WORKDIR /develop/data
WORKDIR /develop/results
WORKDIR /develop/code

COPY .vimrc /root


