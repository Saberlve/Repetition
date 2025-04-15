import json
import re
from transformers import AutoTokenizer

class ICLDatasetBuilder:
    def __init__(self, tokenizer_path=None):
        """
        初始化ICL数据集构建器。
        
        Args:
            tokenizer_path: 可选，分词器路径，用于更精确地提取重复序列
        """
        self.tokenizer = None
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    def extract_repeated_sequence(self, processed_text, original_text):
        """
        从处理后的文本中提取重复序列。
        
        通过比较processed_text和original_text，找出重复序列的第一部分。
        返回重复序列的第一次完整出现。
        """
        # 如果processed_text只是original_text的子串，那么没有重复序列
        if processed_text == original_text or len(processed_text) >= len(original_text):
            return None
        
        # 找出重复序列开始的位置
        common_prefix_length = len(processed_text)
        
        # 在原始文本中寻找重复模式
        remaining_text = original_text[common_prefix_length:]
        
        # 使用正则表达式寻找紧接着的重复模式
        # 假设重复至少有一个有意义的序列（不是单个字符或空格）
        pattern = r'(.{3,}?)\1+'  # 至少3个字符，并且重复一次或多次
        matches = re.search(pattern, remaining_text)
        
        if matches:
            repeated_sequence = matches.group(1)
            return repeated_sequence
        
        # 如果没有找到明显的重复，尝试使用分词器
        if self.tokenizer:
            # 对比处理后文本的最后几个token和原始文本中后续的token
            processed_tokens = self.tokenizer.tokenize(processed_text)
            original_tokens = self.tokenizer.tokenize(original_text)
            
            # 获取处理后文本的最后几个token
            last_few_tokens = processed_tokens[-3:]  # 取最后3个token作为参考
            
            # 在原始文本中查找这些token之后的重复模式
            start_index = len(processed_tokens)
            if start_index >= len(original_tokens):
                return None
            
            # 寻找重复模式
            for sequence_length in range(2, 10):  # 尝试不同长度的序列
                if start_index + sequence_length * 2 <= len(original_tokens):
                    seq1 = original_tokens[start_index:start_index+sequence_length]
                    seq2 = original_tokens[start_index+sequence_length:start_index+sequence_length*2]
                    if seq1 == seq2:
                        return self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(seq1))
        
        return None
    
    def build_icl_prompt(self, repeated_sequence):
        """
        构建ICL提示，将重复序列展示两次。
        """
        if not repeated_sequence:
            return None
        
        # 简单地将重复序列重复两次
        return repeated_sequence + " " + repeated_sequence
    
    def process_natural_prompts(self, natural_prompts_data):
        """
        处理natural prompts数据，构建ICL数据集。
        
        Args:
            natural_prompts_data: 包含natural prompts的列表或文件路径
        
        Returns:
            包含ICL提示的列表
        """
        # 加载数据（如果是文件路径）
        if isinstance(natural_prompts_data, str):
            with open(natural_prompts_data, 'r', encoding='utf-8') as f:
                natural_prompts = json.load(f)
        else:
            natural_prompts = natural_prompts_data
        
        icl_dataset = []
        
        for entry in natural_prompts:
            # 提取重复序列
            repeated_sequence = self.extract_repeated_sequence(
                entry['processed_result'], 
                entry['original_output']
            )
            
            if repeated_sequence:
                # 构建ICL提示
                icl_prompt = self.build_icl_prompt(repeated_sequence)
                
                if icl_prompt:
                    icl_dataset.append({
                        "original_text": entry['original_text'],
                        "repeated_sequence": repeated_sequence,
                        "icl_prompt": icl_prompt
                    })
        
        return icl_dataset
    
    def save_icl_dataset(self, icl_dataset, output_path):
        """
        保存ICL数据集到文件。
        
        Args:
            icl_dataset: ICL数据集
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(icl_dataset, f, indent=4, ensure_ascii=False)


def main(natural_prompts_path, output_path, tokenizer_path=None):
    """
    主函数，处理natural prompts并构建ICL数据集。
    
    Args:
        natural_prompts_path: natural prompts文件路径
        output_path: 输出ICL数据集的文件路径
        tokenizer_path: 分词器路径（可选）
    """
    builder = ICLDatasetBuilder(tokenizer_path)
    icl_dataset = builder.process_natural_prompts(natural_prompts_path)
    builder.save_icl_dataset(icl_dataset, output_path)
    
    print(f"已处理 {len(icl_dataset)} 个有效的ICL提示")
    
    # 输出一些示例
    for i, example in enumerate(icl_dataset[:3]):
        print(f"\n示例 {i+1}:")
        print(f"重复序列: '{example['repeated_sequence']}'")
        print(f"ICL提示: '{example['icl_prompt']}'")


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--natural_prompts_path", default="../dataset/natural_prompts.json", 
                        help="natural prompts文件路径")
    parser.add_argument("--output_path", default="../prompt/icl_dataset.json", 
                        help="输出ICL数据集的文件路径")
    parser.add_argument("--tokenizer_path", default="/disk/disk_20T/wsx/Qwen2-0.5B", 
                        help="分词器路径（可选）")
    
    args = parser.parse_args()
    main(args.natural_prompts_path, args.output_path, args.tokenizer_path)