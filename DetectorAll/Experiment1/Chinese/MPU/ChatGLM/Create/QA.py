from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")
import json
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_curve, auc,
                             precision_score, recall_score, f1_score, confusion_matrix, accuracy_score)


Chinese_model = pipeline('text-classification',
                         model = "/mnt/data132/taozhen/LLMopen_Benchmark/Detector/MPU/Chinese_v2/")


def MPU_method(text):
    result_dict = Chinese_model(text)
    if result_dict and 'score' in result_dict[0]:
        return result_dict
    else:
        raise ValueError("未能获取预期的结果格式")


file_path = "/mnt/data132/taozhen/LLMopen_Benchmark/DatasetFinal/Chinese/ChatGLM/Create/QA.json"
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
# 取出最后1000条数据
last_1000_data = data[-1250:]


# 计算准确率、召回率、精确率、F1
predict_results = []
target_results = []


for item in last_1000_data:
    AI_key = list(item.keys())[-1]  # 获取最后一个键
    AI_text = item[AI_key]  # 获取最后一个键的值
    human_key = list(item.keys())[-2]
    human_text = item[human_key]

    if len(AI_text) > 510:
        AI_text = AI_text[-510:]
    if len(human_text) > 510:
        human_text = human_text[-510:]

    # prediction AI_text
    pre_result1 = MPU_method(AI_text)
    if pre_result1[0]["label"] == "LABEL_0":
        pre_item1 = 0
    else:
        pre_item1 = 1
    predict_results.append(pre_item1)
    target_results.append(1)

    # prediction human_text
    pre_result2 = MPU_method(human_text)
    if pre_result2[0]["label"] == "LABEL_0":
        pre_item2 = 0
    else:
        pre_item2 = 1
    predict_results.append(pre_item2)
    target_results.append(0)


def evaluation(y_truth, y_predict):
    accuracy = accuracy_score(y_truth, y_predict)
    precision = precision_score(y_truth, y_predict)
    recall = recall_score(y_truth, y_predict)
    f1 = f1_score(y_truth, y_predict, average='weighted')

    return accuracy, precision, recall, f1

print(evaluation(target_results, predict_results))

accuracy, precision, recall, f1 = evaluation(target_results, predict_results)

new_result = {
    "Type": "Chinese_MPU_ChatGLM_QA",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}


with open("/mnt/data132/taozhen/LLMopen_Benchmark/DetectorAll/Chinese/Result.txt", 'a', encoding='utf-8') as file:
    json.dump(new_result, file, ensure_ascii=False)
    file.write('\n')

