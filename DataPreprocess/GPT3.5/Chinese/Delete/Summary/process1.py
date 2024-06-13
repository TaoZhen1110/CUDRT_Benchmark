import os
import json
import requests
from multiprocessing import Pool
from tqdm import tqdm

def get_chat_response(input_text):
    data = {
        "model": "gpt-3.5-turbo",  # gpt-4-0125-preview
        "messages": [{"role": "user",
                      "content": input_text
                      }],
        "temperature": 0
    }
    key = ''  # 填入在网页中复制的令牌
    headers = {
        'Authorization': 'Bearer {}'.format(key),
        'Content-Type': 'application/json',
    }
    response = requests.request("POST", "http://47.88.65.188:8300/v1/chat/completions",
                                headers=headers, data=json.dumps(data), timeout=300)
    try:
        content = response.json()['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError):
        content = "1234567"  # or raise an error if you prefer
    return content


def Chinese_Summary(item):
    Human_Content = item['Human_Content']
    Summary_word_number = round(len(item["Human_Abstract"]))

    prompt = f"""
        文本总结是指输入文本后，模型将给定的文本内容概括为简洁、准确的摘要，突出文本的主要信息和关键点。\
        在保留原始文本中最重要、最关键信息的基础上，以更简洁、更易理解的方式呈现文本内容。

        现在我有一篇新闻，需要你帮忙对其进行总结。你的总结需要符合新闻的规范与表达。\
        你的总结字数控制在{Summary_word_number}左右。

        该文本如下：
        {Human_Content}

        最后，你只需要输出总结后的文本即可。
    """
    response_content = get_chat_response(prompt)

    new_item = {
        'ID': item['ID'],
        'Type': "Summary_Chinese_News",
        'Human_Abstract': item['Human_Abstract'],
        'Human_Content': item['Human_Content'],
        'GPT3.5_Content': response_content
    }
    return new_item


def process_line_with_retry(line, max_attempts=3):
    """尝试处理一行数据，最多重试max_attempts次。"""
    for attempt in range(1, max_attempts + 1):
        try:
            return Chinese_Summary(line)
        except Exception as e:
            print(f"处理失败，尝试次数 {attempt}/{max_attempts}: {e}")
            if attempt == max_attempts:
                # 达到最大尝试次数，可以选择返回一个特定的错误标记，或者抛出异常
                return None  # 或者 raise

def write_to_json(data, file_path):
    with open(file_path, 'a', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False)
        file.write('\n')


if __name__ == "__main__":
    processes = 2
    p = Pool(processes=processes)

    origin_data_path = "/home/taoz/AIGC_Detection_Benchmark/Dataset2/Chinese/Delete/Summary/News.json"
    data_path = "/home/taoz/AIGC_Detection_Benchmark/DatasetAll/GPT3.5/Chinese/Delete/Summary/Summary_News.json"

    if not os.path.exists(data_path):
        print("Creating Summary_News.json")
        with open(origin_data_path, 'r', encoding='utf-8') as file:
            json_data = file.read()  # 读取所有行到内存

        data = json.loads(json_data)

        # 使用imap_unordered来获取一个迭代器，这允许我们在任务完成时立即处理结果
        # 使用tqdm显示进度条
        with tqdm(total=len(data)) as progress_bar:
            for result in p.imap_unordered(process_line_with_retry, data):
                if result is not None:
                    write_to_json(result, data_path)
                    progress_bar.update(1)  # 每处理完一行就更新进度条

    else:
        print("Loading Summary_News.json")
        IDfile2_set = set()
        missing_items = []

        with open(origin_data_path, 'r', encoding='utf-8') as file1:
            data1 = json.load(file1)

        with open(data_path, 'r', encoding='utf-8') as file2:
            for line in file2:
                data2 = json.loads(line)
                IDfile2_set.add(data2['ID'])

        for item in data1:
            if item['ID'] not in IDfile2_set:
                missing_items.append(item)

        with tqdm(total=len(missing_items)) as progress_bar:
            for result in p.imap_unordered(process_line_with_retry, missing_items):
                if result is not None:
                    write_to_json(result, data_path)
                    progress_bar.update(1)

    p.close()
    p.join()