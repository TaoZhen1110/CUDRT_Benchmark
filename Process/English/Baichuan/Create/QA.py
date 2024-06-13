import json
import os
import random


all_data = []
id = 0


json_path = "/mnt/data132/taozhen/LLMopen_Benchmark/DatasetAll/Baichuan/English/Create/QA"

for filename in os.listdir(json_path):
    if filename.endswith(".json"):
        file_path = os.path.join(json_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)

                Human_Answer = item['Human_Answer'].replace("\n", "")
                Human_Answer = Human_Answer.replace("\r", "")
                Human_Answer = ' '.join(Human_Answer.split())

                AI_Answer = item['Baichuan_Answer'].replace("\n", "")
                AI_Answer = AI_Answer.replace("\r", "")
                AI_Answer = ' '.join(AI_Answer.split())

                new_data = {
                    'ID': id,
                    'Type': "Baichuan_QA",
                    'Question': item['Question'],
                    'Human_Answer': Human_Answer,
                    'AI_Answer': AI_Answer
                }
                id = id + 1
                all_data.append(new_data)

random.shuffle(all_data)


for i, json_data in enumerate(all_data):
    json_data['ID'] = i           # 重新分配id号，从0开始


with open('/mnt/data132/taozhen/LLMopen_Benchmark/DatasetFinal/English/Baichuan/Create/QA.json', 'w', encoding='utf-8') as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)





