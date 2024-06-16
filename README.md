# CUDRT: Benchmarking the Detection of Human vs. Large Language Models Generated Texts

![CUDRT Benchmark Framework](Images/Fig1.jpg)

## Intoduction
This project focuses on developing a comprehensive bilingual benchmark in both Chinese and English to evaluate the performance of mainstream AI-generated text detectors. Recognizing the challenges in distinguishing between human and AI authorship, our benchmark covers five key operations: Creation, Updating, Deletion, Rewriting, and Translation (CUDRT). By providing extensive datasets for each category and employing the latest mainstream LLMs specific to each language, our benchmark offers a robust evaluation framework. This enables scalable and reproducible experiments, providing critical insights for optimizing AI-generated text detectors and guiding future research directions.

![CUDRT Experiment](Images/Fig8.jpg)

## Installation

Before using CUDRT Benchmark:

1. Ensure you have Python 3.9.0+
2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Project Structure
The project is organized as follows:
```
CUDRT_Benchmark/
├── DataPreprocess/        #
│ ├── Baichuan/
│ ├── ChatGLM/
│ ├── GPT3.5/
│ ├── GPT4Chinese/
│ ├── Llama2English/
│ ├── Llama3English/
│ └── Qwen/
├── Detector/
│ ├── Experiment2/
│ ├── Experiment3/
│ ├── MPU/
│ ├── Roberta/
│ └── XLNet/
├── DetectorAll/
│ ├── Experiment1/
│ ├── Experiment2/
│ └── Experiment3/
├── Process/
│ ├── Chinese/
│ │ ├── Baichuan/
│ │ ├── ChatGLM/
│ │ ├── GPT3.5/
│ │ ├── GPT4/
│ │ └── Qwen/
│ └── English/
│ ├── Baichuan/
│ ├── ChatGLM/
│ ├── GPT3.5/
│ ├── Llama2/
│ ├── Llama3/
│ └── Qwen/
├── DataPreprocess/
├── DatasetAll/
├── DatasetFinal/
```

