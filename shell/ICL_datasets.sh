#!/bin/bash
natural_prompts_path='../dataset/natural_prompts.json'
output_path='../prompt/icl_dataset.json'
tokenizer_path='/disk/disk_20T/wsx/Qwen2-0.5B'

python ../ICL/prepare_ICL_datasets.py \
--natural_prompts_path $natural_prompts_path \
--output_path $output_path \
--tokenizer_path $tokenizer_path