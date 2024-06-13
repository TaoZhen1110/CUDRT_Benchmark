from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated.*")


Chinese_model = pipeline('text-classification', model = "/mnt/data132/taozhen/LLMopen_Benchmark/Detector/MPU/Chinese_v2/")



def MPU_method(text):

    result_dict = Chinese_model(text)
    if result_dict and 'score' in result_dict[0]:
        return result_dict
    else:
        raise ValueError("未能获取预期的结果格式")


text = input("Please input a text:")
result = MPU_method(text)
print(result)



