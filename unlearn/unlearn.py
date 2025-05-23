import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed, Trainer
import transformers
import os
import json
from peft import LoraConfig, get_peft_model,PeftModel
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import copy
import deepspeed
import argparse

def run_unlearn(cfg):
    forget_data_file = cfg.forget_data_path
    retain_data_file = cfg.retain_data_path
    model_path = cfg.model_path
    tokenizer_path = cfg.model_path
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    set_seed(cfg.seed)

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # tokenizer.pad_token = tokenizer.eos_token

    max_length = cfg.max_length


    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code = True)

    # Load reference model for specific loss types
    ref_model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code = True)
    
    
    # Instantiate the forget and retain datasets
    forget_dataset = 
    retain_dataset = 

    dataset = ForgetRetainDataset(
        forget_dataset=forget_dataset,
        retain_dataset=retain_dataset,
    )

    steps_per_epoch = len(dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if True:
        print("enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    config = LoraConfig(
        r=cfg.LoRA_r, 
        lora_alpha=cfg.LoRA_alpha, 
        target_modules=find_all_linear_names(model), 
        lora_dropout=cfg.LoRA_dropout,
        bias="none", 
        task_type="CAUSAL_LM"
    )
    if cfg.LoRA_r != 0:
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, steps_per_epoch),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="steps",
            save_steps=steps_per_epoch,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            deepspeed=cfg.ds_config,
            weight_decay = cfg.weight_decay,
            evaluation_strategy = "no",
            seed=cfg.seed,
            report_to="none", # close wandb
        )

    trainer = IterativeUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset = dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=dataset.get_collate_fn()
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    trainer.train()

    model.save_pretrained(cfg.save_dir)
    tokenizer.save_pretrained(cfg.save_dir)

    # delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                # delete the directory
                import shutil
                shutil.rmtree(global_step_dir)


class IterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 **kwargs):
        self.ref_model = kwargs.pop("ref_model", None)
        self.beta = kwargs.pop("beta", 0.1) 

        super().__init__(*args, **kwargs)
        if self.ref_model is not None:
            # ref_model = ref_model.eval()
            self.ref_model = self.e_prepare_deepspeed(self.ref_model)



    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        # set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False

        return model

    def compute_loss(self, model, x, return_outputs=False, num_items_in_batch=None):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        ### 1. Split the input ###
        
        x_f, x_r = x

        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'],
                                                                                                 dtype=torch.bool)
        )
        with torch.no_grad():
            outputs_f_ref = self.ref_model(
                x_f['input_ids'],
                labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'],
                                                                                                     dtype=torch.bool)
            )

        outputs_f_loss = get_batch_loss(outputs_f.logits, x_f['labels'])
        outputs_f_ref_loss = get_batch_loss(outputs_f_ref.logits, x_f['labels'])
        neg_log_ratio = outputs_f_loss - outputs_f_ref_loss
        loss_npo = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
        loss = loss_npo 

        return (loss, outputs_f) if return_outputs else loss

    # def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
    #     input_ids, labels, attention_mask = x
    #     # forward pass
    #     with torch.no_grad():
    #         outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    #         logits = outputs.logits
    #         loss = outputs.loss
    #     return (loss, logits, labels)


class BaseDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, max_len: int = 4096):
        self.tokenizer = tokenizer
        # if self.tokenizer.pad_token_id is None:
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_len = max_len
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

    def _pad_or_trim(self, input_sequence: torch.Tensor):
        return pad_or_trim_tensor(
            input_sequence,
            target_length=self.max_len,
            padding_value=self.tokenizer.pad_token_id
        )

    def __len__(self):
        return len(self.input_ids)

    def get_collate_fn(self):
        # Base collate function for subclasses to use or override.
        def collate_fn(batch: List[dict]):
            input_ids = torch.stack([b["input_ids"] for b in batch])
            labels = torch.stack([b["labels"] for b in batch])

            output = {
                "input_ids": input_ids,
                "labels": labels,
            }
            if "attention_mask" in batch[0]:
                attention_masks = torch.stack([b["attention_mask"] for b in batch])
                output["attention_mask"] = attention_masks

            return output

        return collate_fn



class ForgetRetainDataset(Dataset):
    def __init__(self, forget_dataset: BaseDataset, retain_dataset: BaseDataset | None = None):
        self.forget_dataset = forget_dataset
        self.retain_exists = retain_dataset is not None
        self.retain_dataset = retain_dataset

    def __getitem__(self, index):
        forget_data = self.forget_dataset[index]
        if self.retain_exists:
            retain_data = self.retain_dataset[index % len(self.retain_dataset)]
            return forget_data, retain_data
        else:
            return forget_data, None

    def __len__(self):
        return len(self.forget_dataset)

    def get_collate_fn(self):
        def collate_fn(batch: List[Tuple[dict, dict]]):
            batch_forget = {
                "input_ids": torch.stack([pair[0]["input_ids"] for pair in batch]),
                "labels": torch.stack([pair[0]["labels"] for pair in batch]),
            }
            if "attention_mask" in batch[0][0]:
                batch_forget["attention_mask"] = torch.stack([pair[0]["attention_mask"] for pair in batch])

            if self.retain_exists:
                batch_retain = {
                    "input_ids": torch.stack([pair[1]["input_ids"] for pair in batch]),
                    "labels": torch.stack([pair[1]["labels"] for pair in batch]),
                }
                if "attention_mask" in batch[0][1]:
                    batch_retain["attention_mask"] = torch.stack([pair[1]["attention_mask"] for pair in batch])
            else:
                batch_retain = None

            return batch_forget, batch_retain

        return collate_fn

def pad_or_trim_tensor(tensor, target_length, padding_value):
    current_length = tensor.size(0)
    
    if current_length < target_length:
        # Padding
        padding_size = target_length - current_length
        padding_tensor = torch.full((padding_size,), padding_value, dtype=tensor.dtype)
        padded_tensor = torch.cat((tensor, padding_tensor))
        return padded_tensor
    
    elif current_length > target_length:
        # Trimming
        trimmed_tensor = tensor[:target_length]
        return trimmed_tensor
    
    else:
        # No change needed
        return tensor

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

def get_batch_loss(logits, labels, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss

@dataclass
class UnlearningConfig:
    forget_data_path: str
    retain_data_path: str
    model_path: str
    tokenizer_path: str
    save_dir: str
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_epochs: int = 5
    lr: float = 2e-5
    max_length: int = 512
    weight_decay: float = 0.01
    seed: int = 42
    ga_ratio: float = 0.5
    gd_ratio: float = 0.3
    gk_ratio: float = 0.0
    LoRA_r: int = 8
    LoRA_alpha: int = 32
    LoRA_dropout: float = 0.05
    ds_config: dict = None

def unlearn(model_path, output_dir, forget_dataset, retain_dataset):
   
    deepspeed_configs = [
        {
            "zero_optimization": {
                "stage": 0,
                "offload_optimizer": {
                    "device": "none",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "none",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto", 
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True
            },
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "gradient_accumulation_steps": "auto",
            "bf16": {
                "enabled": True
            }
        }
    ]
    
    config = UnlearningConfig(
            forget_data_path=forget_dataset,
            retain_data_path=retain_dataset,
            model_path=model_path,
            tokenizer_path=model_path,
            save_dir={output_dir},
            batch_size=1,
            gradient_accumulation_steps=4,
            num_epochs=5,
            lr=1e-4,
            max_length=256,
            weight_decay=0.01,
            seed=42,
            ga_ratio=0.4,
            gd_ratio=0.4,
            gk_ratio=0.2,
            LoRA_r=32,
            LoRA_alpha=32,
            LoRA_dropout=0.05,
            ds_config=deepspeed_configs[0]
        )
    
    run_unlearn(config)
    
   
if __name__ == "__main__":        
    parser = argparse.ArgumentParser()
    parser.add_argument("--forget_dataset", type=str, default="../semeval25-unlearning-data/train/forget.jsonl")
    parser.add_argument("--retain_dataset", type=str, default="../semeval25-unlearning-data/train/retain.jsonl")
    parser.add_argument("--model_path", type=str, default="../../semeval25-unlearning-model")
    parser.add_argument("--output_dir", type=str, default="./output_merge_16")
    args = parser.parse_args()
    
    unlearn(args.model_path, args.output_dir, args.forget_dataset, args.retain_dataset)