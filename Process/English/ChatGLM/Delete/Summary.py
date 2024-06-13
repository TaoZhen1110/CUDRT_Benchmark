import json
import os
import random

all_data = []
id = 0


file_path = "/mnt/data132/taozhen/LLMopen_Benchmark/DatasetAll/ChatGLM/English/Delete/Summary/Summary_News.json"
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)

        Human_Abstract = item['Human_Abstract'].replace("\n", "")
        Human_Abstract = Human_Abstract.replace("\r", "")
        Human_Abstract = ' '.join(Human_Abstract.split())

        AI_Content = item['ChatGLM_Content'].replace("\n", "")
        AI_Content = AI_Content.replace("\r", "")
        AI_Content = ' '.join(AI_Content.split())

        new_data = {
            'ID': id,
            'Type': "ChatGLM_Summary",
            'Human_Abstract': Human_Abstract,
            'AI_Summary': AI_Content
        }
        id = id + 1
        all_data.append(new_data)


random.shuffle(all_data)

for i, json_data in enumerate(all_data):
    json_data['ID'] = i           # 重新分配id号，从0开始

with open('/mnt/data132/taozhen/LLMopen_Benchmark/DatasetFinal/English/ChatGLM/Delete/Summary.json', 'w', encoding='utf-8') as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)