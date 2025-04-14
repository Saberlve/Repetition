import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser

def greedy_gen(model, tokenizer, data):
    generated_texts = []  # 初始化列表
    for item in data:
        input_text = item['text']  
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append({'text':input_text,'output':generated_text})
    
    return generated_texts


def prepare_model(model_name_or_path, device, torch_dtype):

    model =  AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype = torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def prepare_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main(model_name_or_path, data_path, device, torch_dtype):
    data = prepare_dataset(data_path)
    model, tokenizer = prepare_model(model_name_or_path, device, torch_dtype)
    generated_texts = greedy_gen(model, tokenizer, data)
    
    # 将生成的文本保存到文件
    with open('natural_prompt.json', 'w', encoding='utf-8') as f:
        json.dump(generated_texts, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help="The name or path of the pre-trained model")
    parser.add_argument('--data_path', type=str, required=True, help="The path to the input dataset (JSON)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="The device to run the model on (e.g., 'cpu', 'cuda:0')")
    parser.add_argument('--torch_dtype', type=str, default='auto', choices=['auto', 'float16', 'float32'], help="The dtype to load the model weights (e.g., 'auto', 'float16', 'float32')")

    args = parser.parse_args()
    
    main(args.model_name_or_path, args.data_path, args.device, args.torch_dtype)
