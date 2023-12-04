#!/bin/bash
conda create --name R_env --file requirements_R_env.txt

conda create --name pytorch_env --file requirements_pytorch_env.txt
conda activate pytorch_env
pip install comet_ml
pip install asteroid==0.5.2
