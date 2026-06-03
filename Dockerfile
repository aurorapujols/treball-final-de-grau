FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /my-work-dir

# System setup
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Select conda path to install libraries in the environment
ENV PATH=/opt/conda/bin:$PATH

# Create environment and install packages
RUN conda create -y -n gpu_env python=3.10 \
    pytorch=2.5.1 torchvision=0.20.1 torchaudio==2.5.1 pytorch-cuda==12.1 \
    faiss-gpu \
    numpy pandas pillow matplotlib tqdm \
    scikit-learn optuna \
    opencv \
    -c pytorch -c nvidia -c conda-forge --override-channels

# Select path for interpreter
ENV PATH=/opt/conda/envs/gpu_env/bin:$PATH