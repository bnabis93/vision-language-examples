# Docker file from : https://github.com/huggingface/transformers/blob/main/docker/transformers-pytorch-gpu/Dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y git libsndfile1-dev tesseract-ocr espeak-ng python3.9 python3-pip ffmpeg
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Use v4.27-release
ARG REF=v4.27-release
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF
RUN python3 -m pip install --no-cache-dir -e ./transformers[dev-torch,testing,video]

# If set to nothing, will install the latest version
ARG PYTORCH='2.0.0'
# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu117'

RUN [ ${#PYTORCH} -gt 0 ] && VERSION='torch=='$PYTORCH'.*' ||  VERSION='torch'; python3 -m pip install --no-cache-dir -U $VERSION --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN python3 -m pip install --no-cache-dir -U 'torchvision==0.15.1' --extra-index-url https://download.pytorch.org/whl/$CUDA
RUN python3 -m pip install --no-cache-dir -U 'torchaudio==2.0.1' --extra-index-url https://download.pytorch.org/whl/$CUDA

RUN python3 -m pip uninstall -y tensorflow flax

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

WORKDIR /workspace
CMD ["/bin/bash"]