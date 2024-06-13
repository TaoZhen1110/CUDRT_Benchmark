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


def English_Refine(item):
    Human_Content = item['Human_Content']
    Refine_word_number = round(0.8 * len(item["Human_Content"]))

    prompt = f"""
        Text refinement refers to the process where, after inputting text, the model simplifies,\ 
        compresses, or refines the given textual content into a form that is more concise, clear,\ 
        and easy to understand. Under the premise of preserving the original text's main information and meaning,\ 
        the text is made more compact by eliminating redundant information, simplifying sentence structures,\ 
        and distilling key points.

        I currently have a fragment of an academic paper in English that I need your help to refine.\ 
        Your refinement needs to conform to the standards and expressions of academic papers,\ 
        and the refined text should be around {Refine_word_number} letters in length.

        The text is as follows:
        {Human_Content}

        Finally, you only need to output the refined text.
    """
    response_content = get_response(prompt)

    new_item = {
        'ID': item['ID'],
        'Type': "Refine_English_Thesis",
        'Human_Abstract': item['Human_Abstract'],
        'Human_Content': item['Human_Content'],
        'Qwen_Content': response_content
    }
    return new_item


def process_line_with_retry(line, max_attempts=3):
    """尝试处理一行数据，最多重试max_attempts次。"""
    for attempt in range(1, max_attempts + 1):
        try:
            return English_Refine(line)
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

    origin_data_path = "/mnt/data134/taozhen/LLMopen_Benchmark/Dataset2/English/Delete/Refine/Thesis.json"
    data_path = "/mnt/data134/taozhen/LLMopen_Benchmark/DatasetAll/Qwen/English/Delete/Refine/Refine_Thesis.json"

    if not os.path.exists(data_path):
        print("Creating Refine_Thesis.json")
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
        print("Loading Refine_Thesis.json")
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