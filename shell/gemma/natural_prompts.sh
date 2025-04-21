#!/bin/bash
file_path='/mnt/20t/wsx/Repetition/prompt/minipile_train_gemma2.json'
tokenizer_path='/mnt/20t/wsx/models/gemma-2-9b-base'   #same as the generate model
output_path='/mnt/20t/wsx/Repetition/dataset/natural_prompts_gemma2.json'
nohup python ../../ICL/prepare_natural_dataset.py \
--file_path $file_path \
--tokenizer_path $tokenizer_path \
--output_path $output_path | tee ../../logs/natural_prompts.log &