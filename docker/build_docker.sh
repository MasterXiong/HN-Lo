#!/bin/bash
set -x

TAG=zheng/hn_lora
# PARENT=nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
# PARENT=nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04
PARENT=nvidia/cudagl:11.4.2-devel-ubuntu20.04
USER_ID=`id -u`

docker build -f docker/Dockerfile \
  --build-arg PARENT_IMAGE=${PARENT} \
  --build-arg USER_ID=${USER_ID} \
  -t ${TAG} .