# Base Image
ARG PARENT_IMAGE
FROM $PARENT_IMAGE

# Args need to be below FROM!
ARG USER_ID
ARG PYTHON_VERSION=3.10

ENV NVIDIA_DRIVER_CAPABILITIES all

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    ffmpeg \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    # lib for SAPIEN rendering
    libglvnd-dev \
    && rm -rf /var/lib/apt/lists/*

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# install Simpler env
RUN git clone https://github.com/simpler-env/SimplerEnv --recurse-submodules
WORKDIR SimplerEnv
# install dependencies
RUN pip install tensorflow==2.15.0
RUN pip install -r requirements_full_install.txt
RUN pip install tensorflow[and-cuda]==2.15.1
RUN pip install git+https://github.com/nathanrooy/simulated-annealing
# install simpler_env
RUN pip install -e ./ManiSkill2_real2sim
RUN pip install -e .

# install Octo
RUN pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
COPY ./ /user/octo
WORKDIR /user/octo
RUN pip install -e .
# manually upgrade optax to version 0.2.0 to avoid error when creating TrainState
# but may not work with lorax based on the installation warning information
# RUN pip install optax==0.2.0

# install gsutil to download OXE dataset
# Add the Google Cloud SDK distribution URI as a package source
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# Import the Google Cloud public key
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
# Update and install the Google Cloud SDK (includes gsutil)
RUN apt-get update && apt-get install -y google-cloud-sdk

# https://github.com/haosulab/ManiSkill/issues/9
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Change permissions
RUN useradd --shell /bin/bash -u ${USER_ID} -o -d /user user
RUN chown -R user:user /user && chmod -R u+w /user
# RUN chown -R user:user / && chmod -R u+w /

# Set python ENV variables
ENV PYTHONUNBUFFERED=1