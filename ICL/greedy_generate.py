import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from argparse import ArgumentParser
verbose = True
def greedy_gen(model, tokenizer, data):
    generated_texts = []  # 初始化列表
    for item in tqdm(data):
        input_text = item['text']  
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=200,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result ={'text':input_text,'output':generated_text}
        print(result)
        generated_texts.append(result)
    
    return generated_texts


def prepare_model(model_name_or_path, device, torch_dtype):

    model =  AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype = torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def prepare_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def main(model_name_or_path, input_data_path, output_data_path, device, torch_dtype):
    data = prepare_dataset(input_data_path)
    model, tokenizer = prepare_model(model_name_or_path, device, torch_dtype)
    generated_texts = greedy_gen(model, tokenizer, data)
    
    # 将生成的文本保存到文件
    with open(output_data_path, 'w', encoding='utf-8') as f:
        json.dump(generated_texts, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', default='../../Qwen2-0.5B')
    parser.add_argument('--input_data_path', default='../prompt/minipile_train.json')
    parser.add_argument('--output_data_path')
    parser.add_argument('--device',default='cuda' )
    parser.add_argument('--torch_dtype', default='bfloat16')

    args = parser.parse_args()
    
    main(args.model_name_or_path, args.input_data_path, args.output_data_path, args.device, args.torch_dtype)
