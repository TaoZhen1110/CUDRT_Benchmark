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


def English_Complete(item):
    Human_Splitcontent50 = item['Human_Splitcontent_0.50']
    word_number = round(len(item["Human_Content"]) - len(item["Human_Splitcontent_0.50"]))

    prompt = f"""
        Text completion refers to the model's ability to predict subsequent text based on the given partial text,\ 
        ensuring that the completed text logically and contextually aligns with the initial input.

        I now have an incomplete English academic paper that I need help completing. Your completion must adhere to the norms and expressions typical of academic writing.\ 
        Your completion should be around {word_number} letters in length.

        The paper is as follows:
        {Human_Splitcontent50}

        You only need to output the text you complete.
    """
    response_content = get_response(prompt)

    new_item = {
        'ID': item['ID'],
        'Type': "Complete_English_Thesis",
        "Complete_Ratio": 0.5,
        'Human_Content': item['Human_Content'],
        'Qwen_Complete_Content': item["Human_Splitcontent_0.50"] + response_content
    }
    return new_item


def process_line_with_retry(line, max_attempts=3):
    """尝试处理一行数据，最多重试max_attempts次。"""
    for attempt in range(1, max_attempts + 1):
        try:
            return English_Complete(line)
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

    origin_data_path = "/mnt/data134/taozhen/LLMopen_Benchmark/Dataset2/English/Create/Complete/Thesis_ratio50_1.json"
    data_path = "/mnt/data134/taozhen/LLMopen_Benchmark/DatasetAll/Qwen/English/Create/Complete/Complete_Thesis_ratio50.json"

    if not os.path.exists(data_path):
        print("Creating Complete_Thesis_ratio50.json")
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
        print("Loading Complete_Thesis_ratio50.json")
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