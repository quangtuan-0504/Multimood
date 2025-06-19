#!/bin/bash

touch ~/.no_auto_tmux

conda env create -f environment.yml

conda init

source ~/.bashrc

conda activate smes_multimodal

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

##### Run this after activate the environment #####
'
pip uninstall bitsandbytes -y
pip install --upgrade bitsandbytes
pip install deepspeed
pip install h5py
pip install flash-attn --no-build-isolation
pip uninstall deepspeed -y
pip install deepspeed==0.15.4
'