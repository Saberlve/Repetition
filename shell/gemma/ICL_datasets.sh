#!/bin/bash
natural_prompts_path='../../dataset/natural_prompts_gemma2.json'
output_path='../../dataset/icl_dataset_gemma2.json'

python ../../ICL/prepare_ICL_datasets.py \
--natural_prompts_path $natural_prompts_path \
--output_path $output_path 