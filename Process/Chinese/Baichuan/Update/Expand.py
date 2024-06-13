import json
import os
import random


all_data = []
id = 0


json_path = "/mnt/data132/taozhen/LLMopen_Benchmark/DatasetAll/Baichuan/Chinese/Update/Expand"

for filename in os.listdir(json_path):
    if filename.endswith(".json"):
        file_path = os.path.join(json_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)

                Human_Content = item['Human_Content'].replace("\n", "")
                Human_Content = Human_Content.replace("\r", "")
                Human_Content = ' '.join(Human_Content.split())

                AI_Content = item['Baichuan_Content'].replace("\n", "")
                AI_Content = AI_Content.replace("\r", "")
                AI_Content = ' '.join(AI_Content.split())

                new_data = {
                    'ID': id,
                    'Type': "Baichuan_Expand",
                    'Human_Content': Human_Content,
                    'AI_Content': AI_Content
                }
                id = id + 1
                all_data.append(new_data)

random.shuffle(all_data)


for i, json_data in enumerate(all_data):
    json_data['ID'] = i           # 重新分配id号，从0开始


with open('/mnt/data132/taozhen/LLMopen_Benchmark/DatasetFinal/Chinese/Baichuan/Update/Expand.json',
          'w', encoding='utf-8') as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)





