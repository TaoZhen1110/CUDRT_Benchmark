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
    

def English_Complete(item):
    Human_Splitcontent25 = item['Human_Splitcontent_0.25']
    word_number = round(len(item["Human_Content"]) - len(item["Human_Splitcontent_0.25"]))

    prompt = f"""
        Text completion refers to the model's ability to predict subsequent text based on the given partial text,\ 
        ensuring that the completed text logically and contextually aligns with the initial input.

        I have an incomplete news article that I need help completing. Your completion must adhere to the norms and expressions typical of news writing.\ 
        Your completion should be around {word_number} letters in length.

        The news is as follows:
        {Human_Splitcontent25}

        You only need to output the text you complete.
    """
    response_content = get_response(prompt)

    new_item = {
        'ID': item['ID'],
        'Type': "Complete_English_News",
        "Complete_Ratio": 0.75,
        'Human_Content': item['Human_Content'],
        'ChatGLM_Complete_Content': item["Human_Splitcontent_0.25"] + response_content
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
    processes = 5
    p = Pool(processes=processes)

    origin_data_path = "/mnt/data132/taozhen/LLMopen_Benchmark/Dataset2/English/Create/Complete/News_ratio75_1.json"
    data_path = "/mnt/data132/taozhen/LLMopen_Benchmark/DatasetAll/ChatGLM/English/Create/Complete/Complete_News_ratio75.json"

    if not os.path.exists(data_path):
        print("Creating Complete_News_ratio75.json")
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
        print("Loading Complete_News_ratio75.json")
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