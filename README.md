
## 项目工程说明

### 文件结构
├── bert_rumor_model/ # 训练好的BERT模型文件夹
├── BERTmodel.py # BERT模型训练脚本
├── explanation_service.py # LLM解释生成（调用交大API）
├── test.py # 模型测试脚本
├── results.csv # 预测结果
├── results_with_explanation.csv # 预测结果+判断依据
├── requirements.txt # 依赖库列表
└── report.pdf # 大作业报告


### 部署安装

1. **安装Python**（推荐版本 3.11）

2. **安装依赖**
   ```bash
   pip install -r requirements.txt

### 配置 API Key

本项目使用上海交通大学本地大模型 API 生成判断依据。

1. **申请 API Key**：访问 https://claw.sjtu.edu.cn/guide/sjtu-api/，按指引申请。

2. **填写 Key**：打开 `explanation_service.py`，找到以下代码行：
   ```python
   SJTU_API_KEY = ""  

### 确认模型文件

确保 bert_rumor_model/ 文件夹内有 config.json、model.safetensors、tokenizer_config.json

### 模型运行
### 方式一：单条文本预测
 ```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = "./bert_rumor_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

text = "输入你要检测的文本"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()

print(f"预测结果: {pred}（1=谣言，0=非谣言）")

### 方式二：批量测试
   ```python
   python test.py
会生成 results.csv 文件，包含每条文本的预测结果。

### 方式三：生成判断依据
   ```python
  python explanation_service.py
会生成 results_with_explanation.csv 文件，包含预测结果 + 解释文字。

