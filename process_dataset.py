import os
import json



data_path = 'minipile_train.json'
save_path = './repetition_experiment/prompt'
os.makedirs(save_path, exist_ok=True)

with open(data_path, 'r',encoding='utf-8') as f:
    data = json.load(f)
    
save_data=[]
for i, item in enumerate(data):
    if len(item['text']) < 50 : continue
    save_data.append({'text':item['text'][ :50]})
    
save_path = os.path.join(save_path, data_path.split('/')[-1])
with open(save_path, 'w',encoding='utf-8') as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
    