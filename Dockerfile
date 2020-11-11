FROM docker.io/nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV PATH /root/.local/bin:/root/.pyenv/bin:/root/.pyenv/shims:${PATH}

RUN apt-get update \
&& apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

RUN mkdir -p /root/.local/bin \
&& curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash

ENV PYTHONPATH /root/.local/lib:${PYTHONPATH}

RUN pyenv install 3.7.0 \
&& pyenv global 3.7.0

COPY requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /root/.local/lib

WORKDIR /root

COPY vae.py .
COPY trainer.py .
