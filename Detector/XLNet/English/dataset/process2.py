import json
import random

train_data = []
val_data = []

with open("/mnt/data132/taozhen/LLMopen_Benchmark/Detector/Roberta/English/dataset/alldata.json",
          'r', encoding='utf-8') as f:
    data = json.load(f)
    random.shuffle(data)

    for item in data:
        if item["ID"] < 30000:
            train_data.append(item)
        elif item["ID"] >= 30000 and item["ID"] < 38000:
            val_data.append(item)

with open('/mnt/data132/taozhen/LLMopen_Benchmark/Detector/Roberta/English/dataset/train.json',
          'w', encoding='utf-8') as file:
    json.dump(train_data, file, ensure_ascii=False, indent=4)

with open('/mnt/data132/taozhen/LLMopen_Benchmark/Detector/Roberta/English/dataset/val.json',
          'w', encoding='utf-8') as file:
    json.dump(val_data, file, ensure_ascii=False, indent=4)

