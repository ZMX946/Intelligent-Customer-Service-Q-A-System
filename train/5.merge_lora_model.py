# -*- coding: utf-8 -*-
"""
合并 LoRA 权重到基座模型
============================
将 LoRA adapter 合并到基座模型，输出完整的微调模型。

用法：
    python 5.merge_lora_model.py
    python 5.merge_lora_model.py --base ../models/Qwen1.5-1.8B-Chat \
                                  --adapter ./lora-adapter \
                                  --output ./qwen-merged-model

依赖：
    pip install transformers peft torch
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 获取当前脚本所在目录
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)


def merge_lora_and_base(base_model_path: str, lora_model_path: str, save_path: str):
    """
    合并 LoRA 权重到基座模型，并保存成完整模型
    
    参数：
        base_model_path: 基座模型路径（如 Qwen1.5-1.8B）
        lora_model_path: LoRA 适配器路径
        save_path: 合并后模型保存目录
    """
    # 参数校验
    if not os.path.exists(base_model_path):
        raise FileNotFoundError(f"基座模型路径不存在: {base_model_path}")
    if not os.path.exists(lora_model_path):
        raise FileNotFoundError(f"LoRA adapter 路径不存在: {lora_model_path}")
    
    print(f"🔹 加载基座模型: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    print(f"🔹 加载 LoRA 权重: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)

    print("🔹 合并 LoRA 到基座模型...")
    model = model.merge_and_unload()  # 把 LoRA 权重合并到原始权重，并卸载 LoRA 结构

    # 确保输出目录存在
    os.makedirs(save_path, exist_ok=True)
    
    print(f"🔹 保存合并后的模型: {save_path}")
    model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(save_path)

    print(f"✅ 模型已保存到: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到基座模型")
    parser.add_argument("--base", default=os.path.join(_PROJECT_DIR, "models", "Qwen1.5-1.8B-Chat"),
                        help="基座模型路径")
    parser.add_argument("--adapter", default="./lora-adapter",
                        help="LoRA adapter 路径")
    parser.add_argument("--output", default="./merged-model",
                        help="输出目录")
    args = parser.parse_args()

    merge_lora_and_base(args.base, args.adapter, args.output)
