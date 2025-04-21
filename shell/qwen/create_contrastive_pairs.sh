#!/bin/bash
icl_dataset_path='../../dataset/icl_dataset_Qwen2.5.json'
raw_dataset_path='../../raw_dataset/minipile_train.json'
output_path='../../dataset/contrastive_dataset_Qwen2.5.json'

python ../../DPO/generate_datasets.py --icl_dataset_path $icl_dataset_path --raw_dataset_path $raw_dataset_path --output_path $output_path