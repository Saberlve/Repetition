import re
from argparse import ArgumentParser
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
        
        添加验证步骤，确保检测到的重复模式在剩余文本中持续存在。
        
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
                    
                    # 新增：验证该模式是否在剩余文本中持续存在
                    if self.verify_repetition_pattern(remaining_token_ids, i, cycle_length):
                        # 找到了循环重复
                        # 提取前50个字符后到循环开始的文本加上循环的第一个token
                        prefix_text = text[:prompt_end]
                        
                        # 解码从prompt_token_end到重复开始的token
                        middle_text = self.tokenizer.decode(token_ids[prompt_token_end:prompt_token_end+i])
                        
                        # 解码循环的第一个token
                        first_token_text = self.tokenizer.decode([remaining_token_ids[i]])
                        
                        repeated_sequence = self.tokenizer.decode(potential_cycle_ids)
                        return prefix_text + middle_text + first_token_text, repeated_sequence
        
        # 没有找到重复循环
        return None
    
    def verify_repetition_pattern(self, token_ids, start_idx, cycle_length):
        """
        验证检测到的重复模式是否在剩余文本中持续存在。
        
        Args:
            token_ids: 剩余文本的token IDs
            start_idx: 检测到的重复模式开始位置
            cycle_length: 重复模式的长度
            
        Returns:
            bool: 如果重复模式在剩余文本中持续存在则返回True，否则返回False
        """
        pattern = token_ids[start_idx:start_idx+cycle_length]
        
        # 计算在剩余文本中可能完整重复的次数
        remaining_length = len(token_ids) - start_idx
        max_repetitions = remaining_length // cycle_length
        
        # 如果重复次数少于2，不算是有效的重复
        if max_repetitions < 2:
            return False
        
        # 验证每个重复周期是否匹配模式
        for rep in range(1, max_repetitions):
            rep_start = start_idx + rep * cycle_length
            if not all(token_ids[rep_start + j] == pattern[j] for j in range(cycle_length)):
                # 如果找到不匹配，尝试检查是否可能是更长的重复模式
                # 这里简单地返回False，让外层函数尝试不同的起点或周期长度
                return False
                
        # 验证重复模式后的剩余部分不破坏模式
        remainder_start = start_idx + max_repetitions * cycle_length
        remainder_length = len(token_ids) - remainder_start
        
        if remainder_length > 0:
            # 检查剩余部分是否是模式的前缀
            for j in range(remainder_length):
                if token_ids[remainder_start + j] != pattern[j]:
                    # 如果剩余部分不匹配模式的前缀，可能是假重复
                    return False
        
        # 重复模式验证通过
        return True
    
    def process_entry(self, entry):
        """处理单个数据条目。"""
        # 查找输出中的循环重复并截断
        return self.find_repetition_cycle(entry["output"])
    
    def process_data(self, data):
        """处理数据列表，只保留找到重复的结果。"""
        results = []
        for entry in data:
            processed_text, repeated_sequence = self.process_entry(entry) if self.process_entry(entry) else (None, None)
            if processed_text is not None:
                results.append({
                    "original_text": entry["text"],
                    "original_output": entry["output"],
                    "processed_result": processed_text,
                    'repeated_sequence': repeated_sequence
                })
        return results

def main(file_path, tokenizer_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    processor = TextProcessor(tokenizer_path)
    results = processor.process_data(data)

    natural_prompts_path = '../dataset/natural_prompts.json'
    with open(natural_prompts_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--file_path", help='path of the greedy generation file', default='/disk/disk_20T/wsx/Repetition/prompt/minipile_train_greedy_generate.json')
    parser.add_argument("--tokenizer_path", default='/disk/disk_20T/wsx/Qwen2-0.5B')
    
    args = parser.parse_args()
    main(args.file_path, args.tokenizer_path)