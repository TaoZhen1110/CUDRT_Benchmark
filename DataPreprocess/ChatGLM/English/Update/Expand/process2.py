import os
import json
import requests
from multiprocessing import Pool
from tqdm import tqdm





def base_prompt_template() -> str:
    template = """<|system|>
    You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
    <|user|>
    {query}
    <|assistant|>
    """
    return template


def get_response(prompt):

    url = "http://172.27.33.133:8103/generate"      # /mnt/huggingface/models/Llama-3-8B-Instruct/

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
    

def English_Expand(item):
    Human_Content = item['Human_Content']

    prompt = f"""
        Text expansion refers to the process where, given an input text, \
        the model generates new text related to the original content but in greater detail, richness, or with more expressed nuances. \
        While maintaining the meaning of the original text, it adds additional information, descriptions, or content, \
        thereby making the text more comprehensive, vivid, or informative. \
        This approach aims to enhance the original message by providing a deeper or broader perspective without altering its fundamental essence.

        I currently have a fragment of an academic paper in English that I need your help to expand. \
        Your expansion needs to conform to the standards and expressions of academic papers.

        The text is as follows:
        {Human_Content}

        Finally, you only need to output the expanded text.
    """
    response_content = get_response(prompt)

    new_item = {
        'ID': item['ID'],
        'Type': "Expand_English_Thesis",
        'Human_Abstract': item['Human_Abstract'],
        'Human_Content': item['Human_Content'],
        'ChatGLM_Content': response_content
    }
    return new_item


def process_line_with_retry(line, max_attempts=3):
    """尝试处理一行数据，最多重试max_attempts次。"""
    for attempt in range(1, max_attempts + 1):
        try:
            return English_Expand(line)
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

    origin_data_path = "/mnt/data132/taozhen/LLMopen_Benchmark/Dataset2/English/Update/Expand/Thesis.json"
    data_path = "/mnt/data132/taozhen/LLMopen_Benchmark/DatasetAll/ChatGLM/English/Update/Expand/Expand_Thesis.json"

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