import re
from argparse import ArgumentParser

import json
import re

import json
import os
from transformers import AutoTokenizer

class TextProcessor:
    def __init__(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def find_repetition_cycle(self, text):
        """
        在文本中查找循环重复的模式，并返回处理后的文本。
        
        使用分词器将文本转换为token，检测循环重复，然后保留重复前的内容
        加上重复内容的第一个token。
        
        如果没有找到重复，返回None。
        """
        # 先获取前50个字符作为初始提示
        prompt_end = min(50, len(text))
        
        # 对整个文本进行分词
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.encode(text)
        
        # 找到大约对应前50个字符的token位置
        char_count = 0
        prompt_token_end = 0
        for i, token in enumerate(tokens):
            char_count += len(token.replace("##", ""))  # 去掉BERT分词器添加的##前缀
            if char_count >= prompt_end:
                prompt_token_end = i + 1
                break
        
        # 如果剩余token太少，无法检测重复
        if len(tokens) - prompt_token_end < 4:
            return None
        
        # 检测循环重复
        remaining_tokens = tokens[prompt_token_end:]
        remaining_token_ids = token_ids[prompt_token_end:]
        
        for cycle_length in range(2, len(remaining_tokens) // 2 + 1):  # 至少2个token构成循环
            for i in range(len(remaining_tokens) - 2 * cycle_length + 1):
                # 获取可能的循环
                potential_cycle_ids = remaining_token_ids[i:i+cycle_length]
                
                # 检查是否至少重复两次
                if (i + 2*cycle_length <= len(remaining_tokens) and 
                    all(remaining_token_ids[i+j] == remaining_token_ids[i+cycle_length+j] 
                        for j in range(cycle_length))):
                    # 找到了循环重复
                    # 提取前50个字符后到循环开始的文本加上循环的第一个token
                    prefix_text = text[:prompt_end]
                    
                    # 解码从prompt_token_end到重复开始的token
                    middle_text = self.tokenizer.decode(token_ids[prompt_token_end:prompt_token_end+i])
                    
                    # 解码循环的第一个token
                    first_token_text = self.tokenizer.decode([remaining_token_ids[i]])
                    
                    return prefix_text + middle_text + first_token_text
        
        # 没有找到重复循环
        return None
    
    def process_entry(self, entry):
        """处理单个数据条目。"""
        # 查找输出中的循环重复并截断
        return self.find_repetition_cycle(entry["output"])
    
    def process_data(self, data):
        """处理数据列表，只保留找到重复的结果。"""
        results = []
        for entry in data:
            processed_text = self.process_entry(entry)
            if processed_text is not None:
                results.append({
                    "original_text": entry["text"],
                    "original_output": entry["output"],
                    "processed_result": processed_text
                })
        return results


    

def main(file_path, tokenizer_path):
    with open(file_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    processor = TextProcessor(tokenizer_path)
    results = processor.process_data(data)

    # Print results
    # for i, result in enumerate(results):
    #     print(f"Entry {i+1}:")
    #     print(f"Processed Result: {result['processed_result']}")
    #     print()
    natural_prompts_path = '../prompt/natural_prompts.json'
    with open(natural_prompts_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))
        
        
if __name__ =='__main__':
    parser = ArgumentParser()
    parser.add_argument("--file_path", help='path of the greedy generation file',default='/disk/disk_20T/wsx/Repetition/prompt/minipile_train_greedy_generate.json')
    parser.add_argument("--tokenizer_path", default='/disk/disk_20T/wsx/Qwen2-0.5B')
    
    args = parser.parse_args()
    main(args.file_path, args.tokenizer_path)
    
    