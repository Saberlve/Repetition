import os
import json



data_path = 'raw_dataset/minipile_test.json'
save_path = 'prompt'
os.makedirs(save_path, exist_ok=True)

with open(data_path, 'r',encoding='utf-8') as f:
    data = json.load(f)
    
save_data=[]
for i, item in enumerate(data):
    if len(item['text']) < 50 : continue
    save_data.append({'text':item['text'][ :50]})
    
save_path = os.path.join(save_path, data_path.split('/')[-1])
with open(save_path, 'w',encoding='utf-8') as f:
    print(f'truncate {len(save_data)} items to {save_path}')
    json.dump(save_data, f, indent=4, ensure_ascii=False)
    