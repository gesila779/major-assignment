run_pipeline.py 使用说明

1. 功能简介

`run_pipeline.py` 是本项目的整合脚本，负责把谣言分类模型和判断依据生成模块串联起来，实现完整的可解释谣言检测流程。

核心功能：
输入文本 → BERT 分类模型 → 输出预测标签 → 生成判断依据 → 保存或打印结果

其中：
`0` 表示非谣言；
`1` 表示谣言；
`explanation` 表示模型判断依据。

该脚本支持两种运行方式：
（1） 单条文本检测；
（2）批量处理 `val.csv` 并生成结果文件。

2. 依赖文件

运行 `run_pipeline.py` 前，请确保项目目录下至少包含以下文件或目录：

AI_tutorial/
├── run_pipeline.py
├── explanation_service.py
├── val.csv
└── bert_rumor_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    ├── vocab.txt
    └── ...

3. 环境依赖

运行前需要安装以下 Python 库：
bash：
pip install torch transformers pandas tqdm openai numpy

如果还需要运行评估或训练代码，也建议安装：

bash：
pip install scikit-learn


4. 单条文本检测

bash：
python run_pipeline.py --model_path .\bert_rumor_model --text "单条文本信息."

输出内容包括：

输入文本
预测标签
谣言概率
判断依据

 5. 批量处理 val.csv

bash：
python run_pipeline.py --model_path .\bert_rumor_model --input .\val.csv --output .\results_with_explanation.csv

运行完成后，会生成：
results_with_explanation.csv

