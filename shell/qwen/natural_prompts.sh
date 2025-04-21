#!/bin/bash
file_path='/mnt/20t/wsx/Repetition/prompt/minipile_train_Qwen2.5.json'
tokenizer_path='/mnt/20t/wsx/models/Qwen2.5-7B'  #same as the generate model
output_path='/mnt/20t/wsx/Repetition/dataset/natural_prompts_Qwen2.5.json'
nohup python ../../ICL/prepare_natural_dataset.py \
--file_path $file_path \
--tokenizer_path $tokenizer_path \
--output_path $output_path | tee ../../logs/natural_prompts.log &