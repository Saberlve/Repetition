import re
from argparse import ArgumentParser
import json
import os
from tqdm import tqdm
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
        if len(tokens) - prompt_token_end < 6:  # 增加最小token数要求
            return None
        
        # 检测循环重复
        remaining_tokens = tokens[prompt_token_end:]
        remaining_token_ids = token_ids[prompt_token_end:]
        
        # 存储找到的有效重复模式
        found_patterns = []
        
        # 修改：最小周期长度为3，避免像"00"这样的短周期
        for cycle_length in range(3, min(len(remaining_tokens) // 3, 30) + 1):  
            for i in range(len(remaining_tokens) - 3 * cycle_length + 1):
                # 获取可能的循环
                potential_cycle_ids = remaining_token_ids[i:i+cycle_length]
                potential_cycle_text = self.tokenizer.decode(potential_cycle_ids)
                
                # 跳过非常短的文本序列
                if len(potential_cycle_text.strip()) < 5:
                    continue
                
                # 检查是否至少重复三次 (更严格的要求)
                if (i + 3*cycle_length <= len(remaining_tokens) and 
                    all(remaining_token_ids[i+j] == remaining_token_ids[i+cycle_length+j] 
                        for j in range(cycle_length)) and
                    all(remaining_token_ids[i+j] == remaining_token_ids[i+2*cycle_length+j]
                        for j in range(cycle_length))):
                    
                    # 验证该模式是否在剩余文本中持续存在
                    repeat_count, pattern_valid = self.verify_repetition_pattern(
                        remaining_token_ids, i, cycle_length)
                    
                    if pattern_valid and repeat_count >= 3:
                        # 计算重复文本的质量分数
                        pattern_score = self.evaluate_pattern_quality(
                            remaining_token_ids[i:i+cycle_length])
                        
                        # 存储找到的模式
                        found_patterns.append({
                            "start_idx": i,
                            "cycle_length": cycle_length,
                            "repeat_count": repeat_count,
                            "score": pattern_score
                        })
        
        # 如果找到了有效的重复模式，选择最优的一个
        if found_patterns:
            # 根据重复次数、模式长度和质量分数排序
            # 先按照重复次数，再按照模式长度(更长更好)，最后按质量分数
            best_pattern = sorted(
                found_patterns, 
                key=lambda x: (x["repeat_count"], x["cycle_length"], x["score"]), 
                reverse=True
            )[0]
            
            i = best_pattern["start_idx"]
            cycle_length = best_pattern["cycle_length"]
            
            # 提取前50个字符后到循环开始的文本加上循环的第一个token
            prefix_text = text[:prompt_end]
            
            # 解码从prompt_token_end到重复开始的token
            middle_text = self.tokenizer.decode(token_ids[prompt_token_end:prompt_token_end+i])
            
            # 添加完整的第一个循环作为结尾
            first_cycle_text = self.tokenizer.decode(remaining_token_ids[i:i+cycle_length])
            
            # 提取并存储重复序列用于调试
            repeated_sequence = first_cycle_text
            
            return {
                "processed_text": prefix_text + middle_text + first_cycle_text,
                "repeated_sequence": repeated_sequence
            }
        
        # 没有找到重复循环
        return None
    
    def evaluate_pattern_quality(self, pattern_ids):
        """
        评估重复模式的质量。
        
        高质量的模式通常具有更多有意义的词语，更长的文本。
        
        Returns:
            float: 质量分数 (0-1.0)
        """
        # 解码模式
        pattern_text = self.tokenizer.decode(pattern_ids)
        
        # 简单计算:
        # 1. 长度得分 - 鼓励更长的模式
        length_score = min(len(pattern_text) / 50, 1.0)
        
        # 2. 单词得分 - 鼓励包含多个单词的模式
        word_count = len(pattern_text.split())
        word_score = min(word_count / 5, 1.0)
        
        # 3. 复杂度得分 - 鼓励包含多个不同字符的模式
        char_diversity = len(set(pattern_text)) / len(pattern_text) if pattern_text else 0
        diversity_score = char_diversity
        
        # 综合得分
        final_score = (length_score * 0.4 + word_score * 0.4 + diversity_score * 0.2)
        return final_score
    
    def verify_repetition_pattern(self, token_ids, start_idx, cycle_length):
        """
        验证检测到的重复模式是否在剩余文本中持续存在。
        
        Args:
            token_ids: 剩余文本的token IDs
            start_idx: 检测到的重复模式开始位置
            cycle_length: 重复模式的长度
            
        Returns:
            tuple: (repeat_count, is_valid)
                - repeat_count: 模式重复的次数
                - is_valid: 重复模式是否有效
        """
        pattern = token_ids[start_idx:start_idx+cycle_length]
        
        # 计算在剩余文本中可能完整重复的次数
        remaining_length = len(token_ids) - start_idx
        max_repetitions = remaining_length // cycle_length
        
        # 如果重复次数少于3，不算是有效的重复
        if max_repetitions < 3:
            return max_repetitions, False
        
        # 计算实际重复的次数
        actual_repetitions = 1  # 第一次出现已经确认
        
        for rep in range(1, max_repetitions):
            rep_start = start_idx + rep * cycle_length
            
            # 检查是否匹配完整的模式
            full_match = True
            for j in range(cycle_length):
                if rep_start + j >= len(token_ids) or token_ids[rep_start + j] != pattern[j]:
                    full_match = False
                    break
            
            if full_match:
                actual_repetitions += 1
            else:
                # 如果不完全匹配，停止计数
                break
        
        # 如果重复次数太少，认为不是有意义的重复
        if actual_repetitions < 3:
            return actual_repetitions, False
            
        # 检查模式本身的质量
        pattern_text = self.tokenizer.decode(pattern)
        
        # 避免纯数字或单一字符的重复
        if pattern_text.isdigit() or (len(pattern_text.strip()) <= 2):
            return actual_repetitions, False
            
        return actual_repetitions, True
    
    def process_entry(self, entry):
        """处理单个数据条目。"""
        # 查找输出中的循环重复并截断
        result = self.find_repetition_cycle(entry["output"])
        if result is not None:
            return {
                "processed_result": result["processed_text"],
                "repeated_sequence": result["repeated_sequence"]
            }
        return None
    
    def process_data(self, data):
        """处理数据列表，只保留找到重复的结果。"""
        results = []
        for entry in tqdm(data):
            processed_result = self.process_entry(entry)
            if processed_result is not None:
                results.append({
                    "original_text": entry["text"],
                    "original_output": entry["output"],
                    "processed_result": processed_result["processed_result"],
                    "repeated_sequence": processed_result["repeated_sequence"]
                })
        return results

def main(file_path, tokenizer_path ,output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    processor = TextProcessor(tokenizer_path)
    results = processor.process_data(data)


    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))
    print('total len is {}'.format(len(results)))
    print('finished')
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--file_path", help='path of the greedy generation file', default='/disk/disk_20T/wsx/Repetition/prompt/minipile_train_greedy_generate.json')
    parser.add_argument("--tokenizer_path", default='/disk/disk_20T/wsx/Qwen2-0.5B')
    parser.add_argument("--output_path")
    
    args = parser.parse_args()
    main(args.file_path, args.tokenizer_path,args.output_path)