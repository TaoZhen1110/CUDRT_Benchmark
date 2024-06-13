import os
import json
import requests
from multiprocessing import Pool
from tqdm import tqdm






def base_prompt_template() -> str:
    template = """<reserved_195>{query}<reserved_196>"""
    return template


def get_response(prompt):

    url = "http://172.27.33.133:8105/generate"      # /mnt/huggingface/models/Llama-3-8B-Instruct/

    template = base_prompt_template()
    query = template.format(query=prompt)

    payload = json.dumps({
        "prompt": query,
        "temperature": 0.2,
        "max_tokens": 3000,
        "n": 1,
        "stop": ["<|eot_id|>", "<|end_of_text|>", "<|end_header_id|>"],
    })

    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)

    try:
        content = response.json()['text'][0].replace(query, '')
    except (KeyError, IndexError, TypeError):
        content = "1234567"  # or raise an error if you prefer
    return content
    

def Chinese_Expand(item):
    Human_Content = item['Human_SplitContent']

    prompt = f"""
        文本扩写是指输入文本后，模型会根据给定的文本内容来生成与之相关但更加详细、丰富或表达更多细节的新文本。\
        在保持原始文本含义的基础上，增加额外的信息、描述或内容，从而使得文本更加详尽、生动或具有更高的信息量。

        现在我有一篇中文论文，需要你帮忙对其进行扩写。你的扩写需要符合论文的规范与表达。

        该文本如下：
        {Human_Content}

        最后，你只需要输出扩写后的文本即可。
    """
    response_content = get_response(prompt)

    new_item = {
        'ID': item['ID'],
        'Type': "Expand_Chinese_Thesis",
        'Human_Abstract': item['Human_Abstract'],
        'Human_Content': item['Human_SplitContent'],
        'Baichuan_Content': response_content
    }
    return new_item


def process_line_with_retry(line, max_attempts=3):
    """尝试处理一行数据，最多重试max_attempts次。"""
    for attempt in range(1, max_attempts + 1):
        try:
            return Chinese_Expand(line)
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
    processes = 5
    p = Pool(processes=processes)

    origin_data_path = "/mnt/data132/taozhen/LLMopen_Benchmark/Dataset2/Chinese/Update/Expand/Thesis.json"
    data_path = "/mnt/data132/taozhen/LLMopen_Benchmark/DatasetAll/Baichuan/Chinese/Update/Expand/Expand_Thesis.json"

    if not os.path.exists(data_path):
        print("Creating Expand_Thesis.json")
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
        print("Loading Expand_Thesis.json")
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