# ===================== 1. 安装依赖（首次运行需执行） =====================
# !pip install torch transformers pandas scikit-learn tqdm

# ===================== 2. 导入所需库 =====================
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import os

# ===================== 3. 超参数配置（可按需调整） =====================
# 模型配置
MODEL_NAME = 'bert-base-uncased'  # 英文文本用uncased，中文可换bert-base-chinese
MAX_SEQ_LENGTH = 128  # 文本最大截断长度，BERT最大支持512
NUM_LABELS = 2  # 二分类任务：谣言/非谣言

# 训练配置
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5  # BERT微调推荐学习率
WARMUP_RATIO = 0.1  # 学习率预热比例
WEIGHT_DECAY = 0.01  # 权重衰减，防止过拟合

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# 模型保存路径
SAVE_DIR = './bert_rumor_model'
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 4. 加载与预处理数据 =====================
# 读取数据集
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')

# 提取核心列：文本text + 标签label
train_texts = train_df['text'].astype(str).tolist()
train_labels = train_df['label'].astype(int).tolist()
val_texts = val_df['text'].astype(str).tolist()
val_labels = val_df['label'].astype(int).tolist()

# 打印数据基本信息
print(f"\n训练集样本数: {len(train_texts)}")
print(f"验证集样本数: {len(val_texts)}")
print(f"训练集标签分布: {pd.Series(train_labels).value_counts().to_dict()}")
print(f"验证集标签分布: {pd.Series(val_labels).value_counts().to_dict()}")

# ===================== 5. 自定义数据集类 =====================
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

        # 文本编码：BERT所需的input_ids、attention_mask、token_type_ids
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 压缩维度，去掉batch维度
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        token_type_ids = encoding['token_type_ids'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label': torch.tensor(label, dtype=torch.long)
        }

# ===================== 6. 加载Tokenizer与BERT模型 =====================
# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# 加载BERT二分类模型
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    output_attentions=False,
    output_hidden_states=False
)
model.to(DEVICE)

# ===================== 7. 创建数据加载器 =====================
# 构建数据集
train_dataset = RumorDataset(train_texts, train_labels, tokenizer, MAX_SEQ_LENGTH)
val_dataset = RumorDataset(val_texts, val_labels, tokenizer, MAX_SEQ_LENGTH)

# 构建数据加载器
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0  # Windows系统建议设为0，Linux可设为2/4
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ===================== 8. 优化器与学习率调度器配置 =====================
# 计算总训练步数
total_steps = len(train_dataloader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

# 优化器：AdamW（BERT微调推荐）
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# 学习率调度器：线性预热+衰减
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 损失函数：二分类交叉熵
loss_fn = nn.CrossEntropyLoss()

# ===================== 9. 训练函数定义 =====================
def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # 进度条
    progress_bar = tqdm(dataloader, desc='Training', leave=False)

    for batch in progress_bar:
        # 数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 累计损失
        total_loss += loss.item()

        # 保存预测结果
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    # 计算训练集指标
    train_acc = accuracy_score(all_labels, all_preds)
    train_precision = precision_score(all_labels, all_preds, zero_division=0)
    train_recall = recall_score(all_labels, all_preds, zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, zero_division=0)

    return {
        'loss': avg_loss,
        'accuracy': train_acc,
        'precision': train_precision,
        'recall': train_recall,
        'f1': train_f1
    }

# ===================== 10. 评估函数定义 =====================
def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 保存正类概率，用于计算AUC

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating', leave=False)
        for batch in progress_bar:
            # 数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            # 累计损失
            total_loss += loss.item()

            # 保存预测结果
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # 正类概率
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    # 计算验证集全维度指标
    val_acc = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, zero_division=0)
    val_recall = recall_score(all_labels, all_preds, zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    # 计算AUC（二分类核心指标）
    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except:
        val_auc = 0.0

    return {
        'loss': avg_loss,
        'accuracy': val_acc,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'auc': val_auc
    }

# ===================== 11. 主训练循环 =====================
print("\n===================== 开始训练 =====================")
# 记录最优模型
best_f1 = 0.0
best_epoch = 0

for epoch in range(1, EPOCHS + 1):
    print(f"\n===== Epoch {epoch}/{EPOCHS} =====")

    # 训练一轮
    train_metrics = train_epoch(model, train_dataloader, optimizer, scheduler, loss_fn, DEVICE)
    print(f"训练结果: 损失={train_metrics['loss']:.4f}, 准确率={train_metrics['accuracy']:.4f}, F1={train_metrics['f1']:.4f}")

    # 验证一轮
    val_metrics = evaluate(model, val_dataloader, loss_fn, DEVICE)
    print(f"验证结果: 损失={val_metrics['loss']:.4f}, 准确率={val_metrics['accuracy']:.4f}, F1={val_metrics['f1']:.4f}, AUC={val_metrics['auc']:.4f}")

    # 保存最优模型（基于F1值，谣言检测核心指标）
    if val_metrics['f1'] > best_f1:
        best_f1 = val_metrics['f1']
        best_epoch = epoch
        # 保存模型和tokenizer
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)
        print(f"✅ 最优模型已保存，F1值: {best_f1:.4f}")

print(f"\n===================== 训练结束 =====================")
print(f"最优模型来自第{best_epoch}轮，验证集F1值: {best_f1:.4f}")

# ===================== 12. 最终模型评估 =====================
print("\n===================== 最优模型最终评估 =====================")
# 加载最优模型
best_model = BertForSequenceClassification.from_pretrained(SAVE_DIR)
best_model.to(DEVICE)
best_tokenizer = BertTokenizer.from_pretrained(SAVE_DIR)

# 最终评估
final_metrics = evaluate(best_model, val_dataloader, loss_fn, DEVICE)
print(f"最终验证集指标:")
print(f"损失: {final_metrics['loss']:.4f}")
print(f"准确率: {final_metrics['accuracy']:.4f}")
print(f"精确率: {final_metrics['precision']:.4f}")
print(f"召回率: {final_metrics['recall']:.4f}")
print(f"F1值: {final_metrics['f1']:.4f}")
print(f"AUC值: {final_metrics['auc']:.4f}")

# ===================== 13. 模型加载与推理示例 =====================
print("\n===================== 模型推理示例 =====================")
# 加载模型
infer_model = BertForSequenceClassification.from_pretrained(SAVE_DIR)
infer_model.to(DEVICE)
infer_model.eval()
infer_tokenizer = BertTokenizer.from_pretrained(SAVE_DIR)

# 测试文本
test_texts = [
    "Swiss museum confirms it will take on Gurlitt art collection",
    "BREAKING: Anonymous has obtained audio files of Ferguson police",
    "This is a normal news article with no false information"
]

# 推理
for text in test_texts:
    # 文本编码
    encoding = infer_tokenizer(
        text,
        max_length=MAX_SEQ_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    token_type_ids = encoding['token_type_ids'].to(DEVICE)

    # 前向传播
    with torch.no_grad():
        outputs = infer_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        logits = outputs.logits
        prob = torch.softmax(logits, dim=1)[:, 1].item()
        pred = torch.argmax(logits, dim=1).item()

    # 输出结果
    print(f"文本: {text[:50]}...")
    print(f"预测标签: {pred} (1=谣言, 0=非谣言), 谣言概率: {prob:.4f}")
    print("-" * 50)