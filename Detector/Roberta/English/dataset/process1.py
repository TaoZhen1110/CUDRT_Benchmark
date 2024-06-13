import json


id = 0
all_data = []


with open("/mnt/data132/taozhen/LLMopen_Benchmark/Detector/Roberta/English/dataset/all.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if len(data["human_answers"]) > 0 and len(data["chatgpt_answers"]) > 0:

            human_text = data["human_answers"][0].replace("\n", "").replace("\t", "")
            new_data1 = {
                "ID": id,
                "human_text": human_text,
                "label": 0
            }
            all_data.append(new_data1)
            id += 1

            AI_text = data["chatgpt_answers"][0].replace("\n", "").replace("\t", "")
            new_data2 = {
                "ID": id,
                "AI_text": AI_text,
                "label": 1
            }
            all_data.append(new_data2)
            id += 1


with open('/mnt/data132/taozhen/LLMopen_Benchmark/Detector/Roberta/English/dataset/alldata.json',
          'w', encoding='utf-8') as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)


