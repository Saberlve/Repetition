#!/bin/bash

model_name_or_path='../../../Qwen2-0.5B'
data_path='../../prompt/minipile_train_Qwen2.5.json'
device="cuda:3"
torch_dtype="bfloat16"

# Run the Python script and display output in both terminal and log file
nohup python /disk/disk_20T/wsx/Repetition/ICL/greedy_generate.py \
--model_name_or_path $model_name_or_path \
--data_path $data_path \
--device $device \
--torch_dtype $torch_dtype | tee ../../logs/greedy_generate.log &
