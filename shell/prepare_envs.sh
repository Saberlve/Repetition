#!/bin/bash
conda create -n llama-factory python=3.10
conda activate llama-factory
cd ../LLaMA-Factory
pip install -e ".[torch,metrics]"

conda create -n easyedit2 python=3.10
conda activate easyedit2
cd ../EasyEdit
pip install -r requirements_2.txt



