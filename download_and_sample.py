from datasets import load_dataset
import json

def sample_dataset(dataset, n_sample, seed=None):
    """
    从datasets中采样指定数量的数据。
    """
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    sampled_dataset = dataset.select(range(min(n_sample, len(dataset))))
    return sampled_dataset

seed = 42
n_sample_train = 1000
n_sample_test = 100

ds = load_dataset("JeanKaddour/minipile")

train_sampled = sample_dataset(ds['train'], n_sample_train, seed=seed)
test_sampled = sample_dataset(ds['test'], n_sample_test, seed=seed)

# 转换数据集为Python列表
train_list = train_sampled.to_list()
test_list = test_sampled.to_list()

# 使用json.dump保存JSON文件
with open("minipile_train.json", "w", encoding="utf-8") as f:
    json.dump(train_list, f, indent=4, ensure_ascii=False)

with open("minipile_test.json", "w", encoding="utf-8") as f:
    json.dump(test_list, f, indent=4, ensure_ascii=False)