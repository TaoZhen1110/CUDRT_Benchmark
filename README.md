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
├── DataPreprocess/        # Contains text generation code, mainly for various LLMs operations in the text generation process.
│ ├── Baichuan/
│ ├── ChatGLM/
│ ├── GPT3.5/
│ ├── GPT4Chinese/
│ ├── Llama2English/
│ ├── Llama3English/
│ └── Qwen/
├── Detector/             # AI text detector source code.
│ ├── Experiment2/
│ ├── Experiment3/
│ ├── MPU/
│ ├── Roberta/
│ └── XLNet/
├── DetectorAll/          # Three types of detection experiment codes and results.
│ ├── Experiment1/
│ ├── Experiment2/
│ └── Experiment3/
├── Process/              # Make further revisions to the generated AI text.
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
├── Origin_Data/          # Original human text dataset.
├── DatasetAll/           # AI generated AI human parallel dataset.
├── DatasetFinal/         # The final dataset used for detector training and testing.
```

## Citation
```
@article{tao2024cudrt,
  title={CUDRT: Benchmarking the Detection of Human vs. Large Language Models Generated Texts},
  author={Tao, Zhen and Li, Zhiyu and Xi, Dinghao and Xu, Wei},
  journal={arXiv preprint arXiv:2406.09056},
  year={2024}
}
```
