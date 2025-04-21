#!/bin/bash

cd ../../LLaMA-Factory
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/train_lora/gemma2_lora_dpo.yaml
