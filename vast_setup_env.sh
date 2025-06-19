#!/bin/bash

touch ~/.no_auto_tmux

conda env create -f environment.yml

conda init

source ~/.bashrc

conda activate smes_multimodal

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
