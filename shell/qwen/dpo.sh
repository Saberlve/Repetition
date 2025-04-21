#!/bin/bash

cd ../../LLaMA-Factory
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/train_lora/qwen2.5_lora_dpo.yaml
# llamafactory-cli train examples/train_lora/gemma2_lora_dpo.yaml
