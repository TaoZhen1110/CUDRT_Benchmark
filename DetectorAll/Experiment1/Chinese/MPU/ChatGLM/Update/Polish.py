from transformers import pipeline
import json
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_curve, auc,
                             precision_score, recall_score, f1_score, confusion_matrix, accuracy_score)

import re
import numpy as np
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")



## 加载预测标签函数
Chinese_model = pipeline('text-classification',
                         model = "/mnt/data132/taozhen/LLMopen_Benchmark/Detector/MPU/Chinese_v2/")
def MPU_method(text):
    result_dict = Chinese_model(text)
    if result_dict and 'score' in result_dict[0]:
        return result_dict
    else:
        raise ValueError("未能获取预期的结果格式")


# 加载文件
file_path = "/mnt/data132/taozhen/LLMopen_Benchmark/DatasetFinal/Chinese/ChatGLM/Update/Polish.json"
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
# 取出最后1000条数据
last_1000_data = data[-1250:]


## 对于大于512个字的段落进行拆分
def extract_short_text(input_text):
    # 根据句号、问号、感叹号进行分割
    sentences = re.split(r'(?<=[。？！])', input_text)
    short_text = []
    for sentence in sentences:
        short_text.append(sentence)
    return short_text

def split_text(text, max_length=510):
    # 将长于max_length的文本分割成多个部分
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]


def process_texts(texts):
    # 遍历所有文本，按需要进行分割
    processed_texts = []
    for text in texts:
        if len(text) > 510:
            # 如果文本长度超过512，则进行分割
            processed_texts.extend(split_text(text, 510))
        else:
            # 如果文本长度不超过512，直接添加到结果列表
            processed_texts.append(text)
    return processed_texts


def combine_sentences_to_paragraphs(sentences, max_length=510):
    paragraphs = []
    current_paragraph = ""

    for sentence in sentences:
        # 检查加入这个句子后是否超过最大长度
        if len(current_paragraph) + len(sentence) + 1 <= max_length:
            # 如果当前段落为空，直接添加句子
            if current_paragraph:
                # 添加一个空格作为句子分隔
                current_paragraph += sentence
            else:
                current_paragraph = sentence
        else:
            # 如果加入句子会超过最大长度，先将当前段落保存到段落列表中
            paragraphs.append(current_paragraph)
            # 开始一个新的段落
            current_paragraph = sentence

    # 不要忘记添加最后一个段落
    if current_paragraph:
        paragraphs.append(current_paragraph)

    return paragraphs


# 预测文本标签
predict_results = []
target_results = []
for item in last_1000_data:
    # 读取 AI文本和 Human文本
    AI_key = list(item.keys())[-1]  # 获取最后一个键
    AI_text = item[AI_key]  # 获取最后一个键的值
    Human_key = list(item.keys())[-2]
    Human_text = item[Human_key]

    # prediction AI_text
    AI_short_text = process_texts(extract_short_text(AI_text))
    AI_paragraphs = combine_sentences_to_paragraphs(AI_short_text)
    AI_list = np.array([0.0, 0.0])
    for paragraph1 in AI_paragraphs:
        pre_result1 = MPU_method(paragraph1)
        if pre_result1[0]["label"] == "LABEL_0":
            AI_list += np.array([pre_result1[0]["score"], 1-pre_result1[0]["score"]])
        else:
            AI_list += np.array([1-pre_result1[0]["score"], pre_result1[0]["score"]])

    average_AI_list = AI_list/len(AI_paragraphs)
    pre_item1 = np.argmax(average_AI_list)
    predict_results.append(pre_item1)
    target_results.append(1)

    # prediction Human_text
    Human_short_text = process_texts(extract_short_text(Human_text))
    Human_paragraphs = combine_sentences_to_paragraphs(Human_short_text)
    Human_list = np.array([0.0, 0.0])
    for paragraph2 in Human_paragraphs:
        pre_result2 = MPU_method(paragraph2)
        if pre_result2[0]["label"] == "LABEL_0":
            Human_list += np.array([pre_result2[0]["score"], 1-pre_result2[0]["score"]])
        else:
            Human_list += np.array([1-pre_result2[0]["score"], pre_result2[0]["score"]])

    average_Human_list = Human_list/len(Human_paragraphs)
    pre_item2 = np.argmax(average_Human_list)
    predict_results.append(pre_item2)
    target_results.append(0)


# 计算准确率、召回率、精确率、F1
def evaluation(y_truth, y_predict):
    accuracy = accuracy_score(y_truth, y_predict)
    precision = precision_score(y_truth, y_predict)
    recall = recall_score(y_truth, y_predict)
    f1 = f1_score(y_truth, y_predict, average='weighted')

    # kappa=cohen_kappa_score(y_test, y_predict)
    return accuracy, precision, recall, f1

print(evaluation(target_results, predict_results))

accuracy, precision, recall, f1 = evaluation(target_results, predict_results)

new_result = {
    "Type": "Chinese_MPU_ChatGLM_Polish",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}


with open("/mnt/data132/taozhen/LLMopen_Benchmark/DetectorAll/Chinese/Result.txt", 'a', encoding='utf-8') as file:
    json.dump(new_result, file, ensure_ascii=False)
    file.write('\n')

