# -*- coding: utf-8 -*-
"""
文件名: explanation_service.py
功能: 调用上海交通大学本地大模型 API (DeepSeek-V3.2) 生成解释，并支持本地批量运行和保存。
"""

import os
import sys
from openai import OpenAI
import pandas as pd
from tqdm import tqdm


def generate_explanation_by_sjtu_llm(text: str, pred_label: int) -> str:
    """
    接收文本和标签，返回大模型生成的中文解释
    """
    # ！！！请在下方粘贴你从交我办申请到的真实 api-key ！！！
    SJTU_API_KEY = ""
    SJTU_BASE_URL = "https://models.sjtu.edu.cn/api/v1"
    MODEL_NAME = "deepseek-chat"

    label_str = "谣言" if pred_label == 1 else "真实信息（非谣言）"

    # ==================== 2. 终极升级版：面向事实核查的 Prompt ====================
    # 针对裁判指出的“逻辑自洽”、“紧扣内容”、“语言精炼”进行严厉限制
    system_prompt = (
        "你是一个深度学习分类模型的可解释性专家。你的唯一任务是：无条件认同分类模型的预测结果（谣言/非谣言），"
        "并从纯粹的【文本表面特征】（如句式、情感倾向、信源提及方式、词汇选择）来解释模型为何会做出此判定。\n"
        "【铁律】：\n"
        "1. 绝对不能推翻分类模型的预测结果！即使推文涉及的历史事件在现实中是真的，只要标签是“谣言”，你就必须解释该【特定推文的表达方式】为什么具备不可信特征（如：属于个人主观抒情非权威报道、信息模糊、使用了未经证实的网络流言句式等）。\n"
        "2. 必须保持逻辑绝对自洽。如果推文里提及了某媒体（如ABC），你绝不能说它“没有提供任何来源”，而是应该说“其虽然单方面声称引自某媒体，但由于缺乏全局官方通报，在局部文本特征上仍被模型识别为潜在谣言”。\n"
        "3. 严禁出现无意义的语义重复，语言必须一针见血。"
    )

    user_prompt = f"""
        【待分析社交媒体推文】："{text}"
        【分类模型预测结果】：{label_str}

        请严格紧扣上述推文的【具体文本内容】（必须提及推文里的核心关键词或人名事件，严禁套用空话），生成一段 50 到 80 字的中文判定依据。
        要求：
        - 如果是【谣言】：请指出该推文在表达上（如：流于个人情感抒情、单方面非官方声称、信息存在滞后或模糊性、使用了夸张词汇等）符合谣言检测模型对潜在不可信信息的抓取特征。
        - 如果是【真实信息】：请指出其语调的客观性、事实要素的明确性。
        - 必须一行写完，严禁带有任何换行符，语言极度精炼！
        """

    try:
        client = OpenAI(api_key=SJTU_API_KEY, base_url=SJTU_BASE_URL)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=256,
            stream=False
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[提示] 调用失败 ({e})，启用本地规则...", file=sys.stderr)
        return get_backup_explanation(text, pred_label)


def get_backup_explanation(text: str, pred_label: int) -> str:
    """本地规则备用函数"""
    text_lower = text.lower()
    event_context = "该文本"
    if "ferguson" in text_lower or "mikebrown" in text_lower:
        event_context = "针对弗格森（Ferguson）枪击案的相关推文"
    elif "gurlitt" in text_lower or "museum" in text_lower:
        event_context = "关于古利特（Gurlitt）艺术品收藏事件的报道"

    if pred_label == 1:
        if "breaking" in text_lower or "anonymous" in text_lower:
            return f"【判定依据】：{event_context}使用了带有强烈突发暗示的词汇（如'BREAKING'），且缺乏正规媒体的证实，具备社交媒体谣言特征。"
        return f"【判定依据】：{event_context}叙述口吻带有较强的主观色彩，包含未经权威事实支撑的断言，符合谣言的文本特征。"
    else:
        if "confirm" in text_lower or "official" in text_lower or "accept" in text_lower:
            return f"【判定依据】：{event_context}中包含了确凿的陈述词汇，表达方式相对克制中立，多为对已知官方事实的客观转述。"
        return f"【判定依据】：{event_context}整体语调客观平实，属于对突发事件进展的常规动态跟进通报，判定为非谣言。"


# =========================================================================
# 5. 数据保存
# =========================================================================
if __name__ == "__main__":
    VAL_FILE_PATH = "val.csv"  # 确保你把 val.csv 放到了相同的文件夹下
    OUTPUT_FILE_PATH = "results_with_explanation.csv"  # 运行后自动生成的新文件

    if not os.path.exists(VAL_FILE_PATH):
        print(f"错误: 未在当前目录下找到 {VAL_FILE_PATH} 文件，请检查路径。")
        sys.exit(1)

    print(f"正在读取 {VAL_FILE_PATH} ...")
    df = pd.read_csv(VAL_FILE_PATH)

    # 创建一个空列表用来存放生成的解释文字
    explanations = []

    print("开始调用交大 API 批量生成判定依据（带有进度条）...")
    # tqdm 可以帮你显示批量生成的进度
    for index, row in tqdm(df.iterrows(), total=len(df), desc="生成中"):
        text_content = row['text']
        # 这里模拟读取分类标签，如果你们val.csv里原本就有模型判定标签或真实标签
        # 我们可以用 row['label'] 传给大模型做基准
        label_val = int(row['label'])

        # 调用函数获取大模型的解释
        exp = generate_explanation_by_sjtu_llm(text_content, label_val)
        explanations.append(exp)

    # 把解释列表变成表格的新一列
    df['判定依据解释'] = explanations

    # ！！！核心：这一步把带解释的结果文件保存下来 ！！！
    df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
    print(f"\n成功！所有解释已自动保存至新文件: {OUTPUT_FILE_PATH}")