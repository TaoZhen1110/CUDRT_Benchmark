import os
import json
import requests
from multiprocessing import Pool
from tqdm import tqdm

def get_response(prompt):
    url = "http://172.27.33.133:8101/v1/chat/completions"      # /mnt/huggingface/models/Qwen1___5-32B-Chat/

    payload = json.dumps({
        "model": "/mnt/huggingface/models/Qwen1___5-32B-Chat/",
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "temperature": 0.1,
        "repetition_penalty": 1.2,
    })
    response = requests.request("POST", url, data=payload)

    try:
        content = response.json()['choices'][0]['message']['content'].strip()
    except (KeyError, IndexError, TypeError):
        content = "1234567"  # or raise an error if you prefer
    return content


def Chinese_Complete(item):
    Human_Splitcontent75 = item['Human_Splitcontent_0.75']
    word_number = round(len(item["Human_Content"]) - len(item["Human_Splitcontent_0.75"]))

    prompt = f"""
        文本补全是指在输入部分文本后，模型可以根据给定的文本来预测接下来的文本，并尽可能保证补全后的文本符合上下文逻辑和语境。

        现在我有一个不完整的中文论文，需要你帮忙补全。你补全的文本需要符合论文的规范与表达。\
        你的补全的字数控制在{word_number}左右。

        该论文如下：
        {Human_Splitcontent75}

        最后你只需输出你补全的文本即可。
    """
    response_content = get_response(prompt)

    new_item = {
        'ID': item['ID'],
        'Type': "Complete_Chinese_Thesis",
        "Complete_Ratio": 0.25,
        'Human_Content': item['Human_Content'],
        'Qwen_Complete_Content': item["Human_Splitcontent_0.75"] + response_content
    }
    return new_item


def process_line_with_retry(line, max_attempts=3):
    """尝试处理一行数据，最多重试max_attempts次。"""
    for attempt in range(1, max_attempts + 1):
        try:
            return Chinese_Complete(line)
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
    processes = 4
    p = Pool(processes=processes)

    origin_data_path = "/mnt/data134/taozhen/LLMopen_Benchmark/Dataset2/Chinese/Create/Complete/Thesis/Thesis_ratio25.json"
    data_path = "/mnt/data134/taozhen/LLMopen_Benchmark/DatasetAll/Qwen/Chinese/Create/Complete/Complete_Thesis_ratio25.json"

    if not os.path.exists(data_path):
        print("Creating Complete_Thesis_ratio25.json")
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
        print("Loading Complete_Thesis_ratio25.json")
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
                    progress_bar.update(1)  # 每处理完一行就更新进度条

    p.close()
    p.join()