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


def Chinese_Translate(item):
    Human_Content = item['Human_SplitContent']

    prompt = f"""
        文本扩写是指输入文本后，模型会根据给定的文本内容来生成与之相关但更加详细、丰富或表达更多细节的新文本。\
        在保持原始文本含义的基础上，增加额外的信息、描述或内容，从而使得文本更加详尽、生动或具有更高的信息量。

        现在我有一篇中文学术论文，需要你帮忙对其进行扩写。你的扩写需要符合学术论文的规范与表达。

        该文本如下：
        {Human_Content}

        最后，你只需要输出扩写后的文本即可。
    """
    response_content = get_chat_response(prompt)

    new_item = {
        'ID': item['ID'],
        'Type': "Expand_Chinese_Thesis",
        'Human_Abstract': item['Human_Abstract'],
        'Human_Content': item['Human_SplitContent'],
        'GPT4_Content': response_content
    }
    return new_item


def process_line_with_retry(line, max_attempts=3):
    """尝试处理一行数据，最多重试max_attempts次。"""
    for attempt in range(1, max_attempts + 1):
        try:
            return Chinese_Translate(line)
        except Exception as e:
            print(f"处理失败，尝试次数 {attempt}/{max_attempts}: {e}")
            if attempt == max_attempts:
                # 达到最大尝试次数，可以选择返回一个特定的错误标记，或者抛出异常
                return None  # 或者 raise


if __name__ == "__main__":
    processes = 10
    p = Pool(processes=processes)

    origin_data_path = "/home/taoz/AIGC_Detection_Benchmark/Dataset2/Chinese/Update/Expand/Thesis.json"
    data_path = "/home/taoz/AIGC_Detection_Benchmark/DatasetAll/GPT4/Chinese/Update/Expand/Expand_Thesis.json"

    if not os.path.exists(data_path):
        print("Creating Expand_Thesis.json")
        with open(origin_data_path, 'r', encoding='utf-8') as file:
            json_data = file.read()  # 读取所有行到内存

        data = json.loads(json_data)

        # 使用imap_unordered来获取一个迭代器，这允许我们在任务完成时立即处理结果
        # 使用tqdm显示进度条
        with tqdm(total=len(data)) as progress_bar:
            result_iterator = p.imap_unordered(process_line_with_retry, data)
            processed_lines = []
            for result in result_iterator:
                processed_lines.append(result)
                progress_bar.update(1)  # 每处理完一行就更新进度条

        # 将处理后的数据写入新文件，这里过滤掉了处理失败返回None的行
        with open(data_path, 'w', encoding='utf-8') as file:
            # file.writelines(filter(None, processed_lines))
            json.dump(processed_lines, file, ensure_ascii=False, indent=4)

    else:
        print("Loading Expand_Thesis.json")
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