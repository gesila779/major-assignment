# -*- coding: utf-8 -*-
"""
run_pipeline.py
功能：整合 BERT 谣言分类模型 + 判断依据生成模块。
实现：
1. 单条文本：输入 text -> 输出 pred_label + explanation
2. 批量文件：输入 val.csv -> 输出 results_with_explanation.csv
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification


DEFAULT_MODEL_PATH = "./bert_rumor_model"
DEFAULT_INPUT_FILE = "val.csv"
DEFAULT_OUTPUT_FILE = "results_with_explanation.csv"
DEFAULT_MAX_SEQ_LENGTH = 128
DEFAULT_BATCH_SIZE = 32


class TextDataset(Dataset):
    """只用于推理的文本数据集，不要求必须有标签。"""

    def __init__(self, texts: List[str], tokenizer, max_seq_length: int):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        # 有些 BERT tokenizer 会返回 token_type_ids，有些模型不一定需要。
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].flatten()

        return item


class BertRumorClassifier:
    """加载训练好的 BERT 二分类模型，输出 0/1 标签。"""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"未找到模型目录：{model_path}\n"
                f"请确认已经运行训练脚本并生成 bert_rumor_model，或用 --model_path 指定正确路径。"
            )

        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"使用设备: {self.device}")
        print(f"加载模型目录: {self.model_path}")

        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        """批量预测，返回 pred_labels 和 rumor_probs。"""
        dataset = TextDataset(texts, self.tokenizer, self.max_seq_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="BERT模型预测中"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                model_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                if "token_type_ids" in batch:
                    model_inputs["token_type_ids"] = batch["token_type_ids"].to(self.device)

                outputs = self.model(**model_inputs)
                logits = outputs.logits

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                all_probs.extend([float(p) for p in probs])
                all_preds.extend([int(p) for p in preds])

        return all_preds, all_probs

    def predict_one(self, text: str) -> Tuple[int, float]:
        preds, probs = self.predict_batch([text])
        return preds[0], probs[0]


def local_backup_explanation(text: str, pred_label: int) -> str:
    """
    当 explanation_service.py 不可用、openai 库未安装、API key 未配置或接口失败时，使用本地规则解释。
    """
    text_lower = str(text).lower()
    event_context = "该文本"

    if "ferguson" in text_lower or "mikebrown" in text_lower or "mike brown" in text_lower:
        event_context = "该弗格森事件相关推文"
    elif "gurlitt" in text_lower or "museum" in text_lower:
        event_context = "该艺术品收藏相关报道"
    elif "ebola" in text_lower:
        event_context = "该埃博拉相关推文"
    elif "breaking" in text_lower:
        event_context = "该突发消息类推文"

    if pred_label == 1:
        if "breaking" in text_lower or "anonymous" in text_lower or "rumor" in text_lower:
            return f"【判定依据】：{event_context}含有突发爆料或未经核实的表达，信息来源不够明确，容易呈现社交媒体谣言特征。"
        return f"【判定依据】：{event_context}存在较强主观判断或缺少权威信源支撑，事实要素不够完整，因此模型判定为谣言。"
    else:
        if "confirm" in text_lower or "official" in text_lower or "said" in text_lower or "report" in text_lower:
            return f"【判定依据】：{event_context}表述较客观，包含确认、报道或消息来源等事实性线索，因此模型判定为非谣言。"
        return f"【判定依据】：{event_context}整体语气相对平实，未出现明显夸张煽动或未经证实的断言，因此模型判定为非谣言。"


def generate_explanation(text: str, pred_label: int) -> str:
    """
    优先调用 explanation_service.py 中的交大 LLM 解释函数；
    若调用失败，自动降级为本地规则解释。
    """
    try:
        from explanation_service import generate_explanation_by_sjtu_llm
        explanation = generate_explanation_by_sjtu_llm(text, pred_label)
        explanation = str(explanation).replace("\n", " ").strip()
        if explanation:
            return explanation
    except Exception as e:
        print(f"[提示] explanation_service 调用失败，使用本地规则解释。原因：{e}", file=sys.stderr)

    return local_backup_explanation(text, pred_label)


def find_text_column(df: pd.DataFrame, text_col: Optional[str] = None) -> str:
    """自动识别文本列名，也支持命令行手动指定。"""
    if text_col:
        if text_col not in df.columns:
            raise ValueError(f"指定的文本列 {text_col} 不存在。当前列名：{list(df.columns)}")
        return text_col

    candidates = ["text", "content", "tweet", "sentence", "Text", "Content", "Tweet", "文本", "推文"]
    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(f"未找到文本列，请使用 --text_col 指定。当前列名：{list(df.columns)}")


def find_label_column(df: pd.DataFrame, label_col: Optional[str] = None) -> Optional[str]:
    """自动识别标签列；如果没有真实标签，返回 None。"""
    if label_col:
        if label_col not in df.columns:
            raise ValueError(f"指定的标签列 {label_col} 不存在。当前列名：{list(df.columns)}")
        return label_col

    candidates = ["label", "target", "y", "rumor", "is_rumor", "Label", "Target", "标签"]
    for col in candidates:
        if col in df.columns:
            return col

    return None


def run_single_text(args):
    classifier = BertRumorClassifier(
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    pred_label, rumor_prob = classifier.predict_one(args.text)
    explanation = generate_explanation(args.text, pred_label)

    print("\n================ 单条文本检测结果 ================")
    print(f"输入文本: {args.text}")
    print(f"预测标签: {pred_label} （0=非谣言，1=谣言）")
    print(f"谣言概率: {rumor_prob:.4f}")
    print(f"判断依据: {explanation}")


def run_csv(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在：{args.input}")

    df = pd.read_csv(args.input)
    text_col = find_text_column(df, args.text_col)
    label_col = find_label_column(df, args.label_col)

    texts = df[text_col].astype(str).tolist()

    classifier = BertRumorClassifier(
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        device=args.device,
    )

    pred_labels, rumor_probs = classifier.predict_batch(texts)

    explanations = []
    for text, pred_label in tqdm(list(zip(texts, pred_labels)), desc="生成判定依据中"):
        explanations.append(generate_explanation(text, pred_label))

    result_df = pd.DataFrame()
    result_df["text"] = texts

    if label_col is not None:
        result_df["true_label"] = df[label_col].astype(int).tolist()

    result_df["pred_label"] = pred_labels
    result_df["rumor_probability"] = rumor_probs
    result_df["explanation"] = explanations

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    result_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\n================ 批量检测完成 ================")
    print(f"输入文件: {args.input}")
    print(f"文本列: {text_col}")
    if label_col is not None:
        print(f"真实标签列: {label_col}")
    else:
        print("未检测到真实标签列，仅输出预测结果。")
    print(f"输出文件: {args.output}")
    print(result_df.head(5).to_string(index=False))


def build_arg_parser():
    parser = argparse.ArgumentParser(description="BERT谣言检测 + 判断依据生成整合脚本")

    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="训练好的BERT模型目录")
    parser.add_argument("--text", type=str, default=None, help="单条待检测文本")
    parser.add_argument("--input", type=str, default=None, help="待批量检测的CSV文件，如 val.csv")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE, help="输出CSV文件路径")
    parser.add_argument("--text_col", type=str, default=None, help="文本列名；不填则自动识别")
    parser.add_argument("--label_col", type=str, default=None, help="真实标签列名；不填则自动识别")
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_MAX_SEQ_LENGTH, help="BERT最大文本长度")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="批量预测batch size")
    parser.add_argument("--device", type=str, default=None, help="指定设备，如 cuda 或 cpu；不填则自动识别")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.text:
        run_single_text(args)
    elif args.input:
        run_csv(args)
    else:
        parser.print_help()
        raise ValueError("请提供 --text 或 --input 参数。")


if __name__ == "__main__":
    main()
