import json
from argparse import ArgumentParser

def process_data(data):
    """
    处理字典列表中的数据，为每个条目添加 'icl_prompts' 键，
    其值为 'repeated_sequence' 重复两次的结果。
    
    :param data: 包含字典的列表
    :return: 更新后的字典列表
    """
    for entry in data:
        # 确保 repeated_sequence 存在
        if "repeated_sequence" in entry:
            repeated_sequence = entry["repeated_sequence"]
            # 将 repeated_sequence 重复两次作为 icl_prompts
            entry["icl_prompts"] = repeated_sequence * 2
        else:
            # 如果缺少 repeated_sequence，设置 icl_prompts 为空字符串
            entry["icl_prompts"] = ""
    return data

def main(input_path, output_path):
    """
    主函数：读取输入文件，处理数据，并保存到输出文件。
    
    :param input_path: 输入文件路径 (JSON 格式)
    :param output_path: 输出文件路径 (JSON 格式)
    """
    try:
        # 读取输入文件
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 检查数据是否为列表
        if not isinstance(data, list):
            raise ValueError("输入文件的内容必须是列表格式")
        
        # 处理数据
        processed_data = process_data(data)
        
        # 写入输出文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
        
        print(f"处理完成，结果已保存到 {output_path}")
    
    except FileNotFoundError:
        print(f"错误：文件 {input_path} 未找到")
    except json.JSONDecodeError:
        print(f"错误：文件 {input_path} 不是有效的 JSON 格式")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--natural_prompts_path", default="../dataset/natural_prompts.json", 
                        help="natural prompts 文件路径")
    parser.add_argument("--output_path", default="../dataset/icl_dataset.json", 
                        help="输出 ICL 数据集的文件路径")
    
    args = parser.parse_args()
    main(args.natural_prompts_path, args.output_path)