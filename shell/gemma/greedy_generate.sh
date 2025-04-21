#!/bin/bash

model_name_or_path='/mnt/20t/wsx/models/gemma-2-9b-base'
input_data_path='../../prompt/minipile_train_sample.json'
output_data_path='../../prompt/minipile_train_gemma2.json'
device="cuda:0"
torch_dtype="bfloat16"

# Run the Python script and display output in both terminal and log file
CUDA_VISIBLE_DEVICES=0 nohup python ../../ICL/greedy_generate.py \
--model_name_or_path $model_name_or_path \
--input_data_path $input_data_path \
--output_data_path $output_data_path \
--device $device \
--torch_dtype $torch_dtype | tee ../../logs/greedy_generate.log &
