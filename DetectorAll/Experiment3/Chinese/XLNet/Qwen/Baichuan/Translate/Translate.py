import json
import argparse   #用于解析命令行参数
import os
import torch
import torch.nn as nn
from sklearn.metrics import (precision_recall_curve, average_precision_score, roc_curve, auc,
                             precision_score, recall_score, f1_score, confusion_matrix, accuracy_score)

from transformers import XLNetTokenizer
import sys
sys.path.append('/mnt/data132/taozhen/LLMopen_Benchmark/DetectorAll/Experiment3/Chinese/XLNet/Qwen/')
from dataprepare import *
from model import XLNet_model


def get_arguments():
    parser = argparse.ArgumentParser()   #创建解析器,ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息

    parser.add_argument('-gpu', type=str, default='2,3')

    ##############   pretrained model setting
    parser.add_argument("-checkpoint", type=str,
                        default='/mnt/data132/taozhen/LLMopen_Benchmark/Detector/XLNet/Chinese/xlnet_pretrained/')
    parser.add_argument("-model_path", type=str,
                        default='/mnt/data132/taozhen/LLMopen_Benchmark/Detector/Experiment3/XLNet/Chinese/Qwen/run_text/run_0/qwen_xlnet_ch.pth')
    return parser.parse_args()


def evaluation(y_truth, y_predict):
    accuracy = accuracy_score(y_truth, y_predict)
    precision = precision_score(y_truth, y_predict)
    recall = recall_score(y_truth, y_predict)
    f1 = f1_score(y_truth, y_predict, average='weighted')

    return accuracy, precision, recall, f1


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    ###################  Dataset prepare #################
    tokenizer = XLNetTokenizer.from_pretrained(args.checkpoint)
    with open("/mnt/data132/taozhen/LLMopen_Benchmark/DatasetFinal/Chinese/Baichuan/Translate/Translate.json", 'r') as f:
        test_data = json.load(f)
    last_test_data = test_data[-1250:]

    testloader = create_combined_dataloader(dataset1 = datapre1(jsondata = last_test_data, tokenizer=tokenizer),
                                            dataset2 = datapre2(jsondata = last_test_data, tokenizer=tokenizer),
                                            batch_size=1, shuffle=False)

    ##加载模型
    model = XLNet_model(checkpoint=args.checkpoint)
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.model_path))

    ## 计算分类结果
    real_labels = []
    pre_labels = []
    model.eval()
    for step, sample_batched in enumerate(testloader):
        input_ids, attention_mask, token_type_ids, real_label = (x.cuda() for x in sample_batched)
        with torch.no_grad():
            outputs = model(enc_inputs=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pre_label = torch.argmax(outputs, 1)
        pre_labels.append(pre_label.cpu().numpy())
        real_labels.append(real_label.cpu().numpy())

    accuracy, precision, recall, f1 = evaluation(real_labels, pre_labels)

    new_result = {
        "Type": "Chinese_XLNet_Qwen_Baichuan_Translate",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    print(new_result)
    with open("/mnt/data132/taozhen/LLMopen_Benchmark/DetectorAll/Experiment3/Chinese/Result.txt", 'a', encoding='utf-8') as file:
        json.dump(new_result, file, ensure_ascii=False)
        file.write('\n')


if __name__ == "__main__":
    args = get_arguments()
    main(args)