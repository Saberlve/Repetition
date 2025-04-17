#!/bin/bash
cd ../EasyEdit
model_name_or_path=../../Qwen2-0.5B
torch_dtype=bfloat16
device=cuda:1


steer_vector_output_dir="steer/vectors/Qwen2-0.5B"
steer_vector_load_dir=("steer/vectors/Qwen2-0.5B/repetition/caa_vector")
generation_output_dir=("steer/generation/Qwen2-0.5B/")

nohup python ../EasyEdit/steering.py \
        model_name_or_path=$model_name_or_path \
        torch_dtype=$torch_dtype \
        device=$device \
        steer_vector_output_dir=$steer_vector_output_dir \
        steer_vector_load_dir=[$steer_vector_load_dir] \
        generation_output_dir=$generation_output_dir | tee -a ../logs/steer_log.txt
