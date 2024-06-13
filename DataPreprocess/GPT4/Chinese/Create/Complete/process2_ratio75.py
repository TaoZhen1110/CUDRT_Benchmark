import os
import json
import requests
from multiprocessing import Pool
from tqdm import tqdm


def get_chat_response(input_text):
    data = {
        "model": "gpt-4-1106-preview",  # gpt-4-0125-preview
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
        content = "123"  # or raise an error if you prefer
    return content


def Chinese_Complete(item):
    Human_Splitcontent25 = item['Human_Splitcontent_0.25']
    word_number = round(len(item["Human_Content"]) - len(item["Human_Splitcontent_0.25"]))

    prompt = f"""
        文本补全是指在输入部分文本后，模型可以根据给定的文本来预测接下来的文本，并尽可能保证补全后的文本符合上下文逻辑和语境。

        现在我有一个不完整的中文论文，需要你帮忙补全。你补全的文本需要符合论文的规范与表达。\
        你的补全的字数控制在{word_number}左右。

        该论文如下：
        {Human_Splitcontent25}

        最后你只需输出你补全的文本即可。
    """
    response_content = get_chat_response(prompt)

    new_item = {
        'ID': item['ID'],
        'Type': "Complete_Chinese_Thesis",
        "Complete_Ratio": 0.75,
        'Human_Content': item['Human_Content'],
        'GPT4_Complete_Content': item["Human_Splitcontent_0.25"] + response_content
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
    processes = 50
    p = Pool(processes=processes)

    origin_data_path = "/home/taoz/AIGC_Detection_Benchmark/Dataset2/Chinese/Create/Complete/Thesis/Thesis_ratio75.json"
    data_path = "/home/taoz/AIGC_Detection_Benchmark/DatasetAll/GPT4/Chinese/Create/Complete/Complete_Thesis_ratio75.json"

    if not os.path.exists(data_path):
        print("Creating Complete_Thesis_ratio75.json")
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
        print("Loading Complete_Thesis_ratio75.json")
        with open(origin_data_path, 'r', encoding='utf-8') as file1:
            data1 = json.load(file1)
        with open(data_path, 'w', encoding='utf-8') as file2:
            data2 = json.load(file2)

        missing_items = []

        ids_file2 = {item['ID'] for item in data2}
        for item in data1:
            if item['ID'] not in ids_file2:
                missing_items.append(item)

        with tqdm(total=len(missing_items)) as progress_bar:
            result_iterator = p.imap_unordered(process_line_with_retry, missing_items)
            processed_lines = []
            for result in result_iterator:
                processed_lines.append(result)
                progress_bar.update(1)  # 每处理完一行就更新进度条

        data2.extend(processed_lines)
        with open(data_path, 'w', encoding='utf-8') as file:
            # file.writelines(filter(None, processed_lines))
            json.dump(data2, file, ensure_ascii=False, indent=4)

    p.close()
    p.join()