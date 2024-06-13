import json

with open("/mnt/data132/taozhen/LLMopen_Benchmark/Dataset2/Chinese/Create/Complete/News_ratio25.json",
          "r", encoding='utf-8') as f:
    data1 = json.load(f)

data_dict = {}
for item in data1:
    data_dict[item['ID']] = item

all_data = []

with open("/mnt/data132/taozhen/LLMopen_Benchmark/DatasetAll/Baichuan/Chinese/Create/Complete/Complete_News_ratio25.json",
          "r", encoding='utf-8') as f:
    for data2 in f:
        item = json.loads(data2)
        if item['ID'] in data_dict:
            Needcomplete_Content = item["Human_Content"].replace(data_dict[item['ID']]["Human_Splitcontent_0.75"], "")
            AI_Content = item["Baichuan_Complete_Content"].replace(data_dict[item['ID']]["Human_Splitcontent_0.75"], "")
            new_item = {
                "ID": item["ID"],
                "Complete_Ratio": item["Complete_Ratio"],
                "Human_Content": item["Human_Content"],
                "Split_Content": data_dict[item['ID']]["Human_Splitcontent_0.75"],
                "Needcomplete_Content": Needcomplete_Content,
                "AI_Content": AI_Content
            }
            all_data.append(new_item)

with open('/mnt/data132/taozhen/LLMopen_Benchmark/DatasetAll/Baichuan/Chinese/Create/Complete/Complete_News_ratio25_1.json',
          'w', encoding='utf-8') as file:
    json.dump(all_data, file, ensure_ascii=False, indent=4)