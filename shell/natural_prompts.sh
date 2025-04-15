#!/bin/bash
file_path='/disk/disk_20T/wsx/Repetition/prompt/minipile_train_greedy_generate.json'
tokenizer_path='/disk/disk_20T/wsx/Qwen2-0.5B'   #same as the generate model

nohup python /disk/disk_20T/wsx/Repetition/ICL/prepare_ICL_dataset.py \
--file_path $file_path \
--tokenizer_path $tokenizer_path | tee ../logs/natural_prompts.log &