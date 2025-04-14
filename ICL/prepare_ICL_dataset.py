import re

def find_repeated_sequences(text):
    """
    查找并返回文本中的重复部分。
    假设重复部分是两个相同的子串。
    
    Args:
        text (str): 输入的文本。
        
    Returns:
        repeated_part (str): 重复的子串。如果没有找到重复，则返回None。
    """
    # 使用正则表达式来查找重复部分
    # 假设重复部分是连续出现的相同子字符串
    pattern = r'(\b\w+\b(?: \b\w+\b)+)(?=.*\1)'
    matches = re.findall(pattern, text)

    if matches:
        # 返回第一个找到的重复部分
        return matches[0]
    return None

def build_icl_prompts(generated_texts):
    """
    基于模型生成的文本，构建In-Context Learning（ICL）提示。
    对重复部分进行标记并重复两次。

    Args:
        generated_texts (list): 模型生成的文本列表，每个元素是一个字符串。

    Returns:
        list: 构建的ICL提示列表。
    """
    icl_prompts = []
    for text in generated_texts:
        repeated_part = find_repeated_sequences(text)
        if repeated_part:
            # 如果找到了重复部分，将其重复两次
            new_prompt = text + ' ' + repeated_part
            icl_prompts.append(new_prompt)
        else:
            icl_prompts.append(text)  # 如果没有找到重复部分，直接添加原始文本
    return icl_prompts

# 示例文本
generated_texts = [
    "iridium complexes. The ruthenium complexes are more efficient catalysts than the iridium complexes.",
    "The ruthenium complexes are more efficient catalysts than the iridium complexes. The ruthenium complexes are more efficient catalysts than the."
]

# 生成ICL提示
icl_prompts = build_icl_prompts(generated_texts)

# 打印ICL提示
for prompt in icl_prompts:
    print(prompt)
