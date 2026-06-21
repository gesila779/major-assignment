# ===================== 谣言模型测试 + 纯数字标签结果表格 =====================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# ===================== 固定配置 =====================
MODEL_PATH = './bert_rumor_model'  # 训练好的模型路径
VAL_FILE = 'val.csv'               # 测试文件
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ===================== 加载模型 & 分词器 =====================
print(f"使用设备: {DEVICE}")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()  # 评估模式

# ===================== 数据集类 =====================
class RumorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_seq_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ===================== 加载验证集数据 =====================
val_df = pd.read_csv(VAL_FILE)
val_texts = val_df['text'].astype(str).tolist()
val_labels = val_df['label'].astype(int).tolist()

# 构建数据加载器
val_dataset = RumorDataset(val_texts, val_labels, tokenizer, MAX_SEQ_LENGTH)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ===================== 模型测试 =====================
all_preds = []  # 模型预测标签（0/1）
all_trues = []  # 真实标签（0/1）

with torch.no_grad():
    for batch in tqdm(val_dataloader, desc="模型测试中"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        true_labels = batch['label'].numpy()

        # 模型预测
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_labels = torch.argmax(outputs.logits, dim=1).cpu().numpy()

        all_preds.extend(pred_labels)
        all_trues.extend(true_labels)

# ===================== 生成结果表格（纯数字0/1，与原数据集一致） =====================
result_df = pd.DataFrame({
    "输入的文本": val_texts,
    "模型判断(0/1)": all_preds,
    "实际标签(0/1)": all_trues
})

# 控制台预览前20行
print("\n" + "="*80)
print("📊 测试结果表格（前20行预览 | 0/1与原数据集完全一致）")
print("="*80)
print(result_df.head(20).to_string(index=False))

# 保存完整表格到CSV
result_df.to_csv("results.csv", index=False, encoding="utf-8-sig")

# ===================== 模型指标输出 =====================
acc = accuracy_score(all_trues, all_preds)
f1 = f1_score(all_trues, all_preds)
print("\n" + "="*80)
print(f"📈 模型综合指标 | 准确率: {acc:.4f} | F1分数: {f1:.4f}")
print(f"✅ 完整结果表格已保存至：results.csv")