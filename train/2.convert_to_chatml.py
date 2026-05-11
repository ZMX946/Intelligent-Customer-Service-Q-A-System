"""
Alpaca → ChatML 格式转换脚本
====================================
输入（Alpaca 格式，本项目生成的训练数据）：
  [
    {
      "instruction": "系统提示词...",
      "input": "用户消息（可能包含【系统检索结果】）",
      "output": "客服回复"
    },
    ...
  ]

输出（ChatML 格式，两种可选）：

  格式 A：messages 列表（HuggingFace / LLaMA-Factory 标准）
  [
    {
      "messages": [
        {"role": "system",    "content": "系统提示词"},
        {"role": "user",      "content": "用户消息"},
        {"role": "assistant", "content": "客服回复"}
      ]
    },
    ...
  ]

  格式 B：对话文本（原始 ChatML token 格式，用于直接 tokenize）
  [
    {
      "text": "<|im_start|>system\\n系统提示词<|im_end|>\\n<|im_start|>user\\n用户消息<|im_end|>\\n<|im_start|>assistant\\n客服回复<|im_end|>"
    },
    ...
  ]

用法：
    # 默认输出 messages 格式（推荐，LLaMA-Factory 直接支持）
    python convert_to_chatml.py --input customerservice_alpaca.json

    # 输出 ChatML text 格式
    python convert_to_chatml.py --input customerservice_alpaca.json --format text

    # 指定输出路径
    python convert_to_chatml.py --input customerservice_alpaca.json --output my_chatml.json

    # 同时输出两种格式
    python convert_to_chatml.py --input customerservice_alpaca.json --format both
"""

import json
import argparse
import logging
from pathlib import Path
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── 格式转换核心函数 ─────────────────────────────────────────────────────────

def alpaca_to_messages(item: dict) -> dict | None:
    """
    转换为 messages 列表格式（HuggingFace / LLaMA-Factory 标准）
    
    注意：input 字段可能包含 RAG 检索结果前缀，整体作为 user 消息内容。
    """
    instruction = item.get("instruction", "").strip()
    user_input  = item.get("input", "").strip()
    output      = item.get("output", "").strip()

    if not user_input or not output:
        return None

    messages = []

    # system 消息（如果有 instruction）
    if instruction:
        messages.append({"role": "system", "content": instruction})

    # user 消息
    messages.append({"role": "user", "content": user_input})

    # assistant 消息
    messages.append({"role": "assistant", "content": output})

    return {"messages": messages}


def alpaca_to_chatml_text(item: dict) -> dict | None:
    """
    转换为原始 ChatML token 文本格式。
    适用于需要自己控制 tokenize 的场景。
    """
    instruction = item.get("instruction", "").strip()
    user_input  = item.get("input", "").strip()
    output      = item.get("output", "").strip()

    if not user_input or not output:
        return None

    parts = []
    if instruction:
        parts.append(f"<|im_start|>system\n{instruction}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_input}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n{output}<|im_end|>")

    return {"text": "\n".join(parts)}


# ─── 统计与验证 ───────────────────────────────────────────────────────────────

def analyze(alpaca_data: list[dict]) -> None:
    """打印输入数据的基本统计"""
    total = len(alpaca_data)
    rag_count = sum(1 for d in alpaca_data if "【系统检索结果】" in d.get("input", ""))
    
    # 输出长度分布
    out_lens = [len(d.get("output", "")) for d in alpaca_data]
    avg_out  = sum(out_lens) / len(out_lens) if out_lens else 0
    max_out  = max(out_lens) if out_lens else 0

    # input 长度分布
    in_lens = [len(d.get("input", "")) for d in alpaca_data]
    avg_in  = sum(in_lens) / len(in_lens) if in_lens else 0
    max_in  = max(in_lens) if in_lens else 0

    log.info("\n════════════════ 输入数据分析 ════════════════")
    log.info(f"  总条数:          {total}")
    log.info(f"  RAG 样本数:      {rag_count}  ({rag_count/total:.1%})")
    log.info(f"  普通样本数:      {total - rag_count}  ({(total-rag_count)/total:.1%})")
    log.info(f"  input 平均长度:  {avg_in:.0f} 字符  最大: {max_in}")
    log.info(f"  output 平均长度: {avg_out:.0f} 字符  最大: {max_out}")

    # 检查字段完整性
    missing_instruction = sum(1 for d in alpaca_data if not d.get("instruction"))
    missing_input       = sum(1 for d in alpaca_data if not d.get("input"))
    missing_output      = sum(1 for d in alpaca_data if not d.get("output"))
    if missing_instruction or missing_input or missing_output:
        log.warning(f"  ⚠️  缺失字段: instruction={missing_instruction}, "
                    f"input={missing_input}, output={missing_output}")
    else:
        log.info("  ✅ 所有字段完整")


def validate_messages(data: list[dict]) -> None:
    """验证转换后的 messages 格式"""
    errors = 0
    for i, item in enumerate(data):
        msgs = item.get("messages", [])
        roles = [m["role"] for m in msgs]
        # 检查顺序正确性
        if not roles or roles[-1] != "assistant":
            log.warning(f"  第 {i} 条：最后一条不是 assistant，roles={roles}")
            errors += 1
        if "user" not in roles:
            log.warning(f"  第 {i} 条：缺少 user 消息")
            errors += 1
    if errors == 0:
        log.info("  ✅ messages 格式验证通过")
    else:
        log.warning(f"  ⚠️  发现 {errors} 条格式异常")


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alpaca → ChatML 格式转换")
    parser.add_argument("--input",  "-i", required=True,
                        help="输入文件路径（Alpaca 格式 JSON）")
    parser.add_argument("--output", "-o", default=None,
                        help="输出文件路径（默认：输入文件名加 _chatml 后缀）")
    parser.add_argument("--format", "-f",
                        choices=["messages", "text", "both"],
                        default="messages",
                        help="输出格式：messages（默认）/ text / both")
    args = parser.parse_args()

    # ── 读取输入 ─────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        log.error(f"文件不存在: {input_path}")
        return

    with open(input_path, encoding="utf-8") as f:
        alpaca_data = json.load(f)

    log.info(f"已读取: {input_path}  ({len(alpaca_data)} 条)")
    analyze(alpaca_data)

    # ── 确定输出路径 ──────────────────────────────────────────────────────────
    stem = input_path.stem  # 不含扩展名的文件名

    def out_path(suffix: str) -> Path:
        if args.output and args.format != "both":
            return Path(args.output)
        return input_path.parent / f"{stem}_{suffix}.json"

    # ── 转换 ─────────────────────────────────────────────────────────────────
    skipped = 0

    if args.format in ("messages", "both"):
        converted = []
        for item in alpaca_data:
            result = alpaca_to_messages(item)
            if result:
                converted.append(result)
            else:
                skipped += 1

        path_m = out_path("chatml_messages")
        with open(path_m, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False, indent=2)

        log.info(f"\n════════════════ messages 格式 ════════════════")
        log.info(f"  转换成功: {len(converted)} 条")
        if skipped:
            log.warning(f"  跳过(字段缺失): {skipped} 条")
        log.info(f"  输出文件: {path_m}  ({path_m.stat().st_size/1024:.1f} KB)")
        validate_messages(converted)

        # 样本预览
        log.info("\n  样本预览（前2条）：")
        for i, item in enumerate(converted[:2], 1):
            log.info(f"\n  [{i}]")
            for msg in item["messages"]:
                role    = msg["role"]
                content = msg["content"]
                preview = content[:60] + "..." if len(content) > 60 else content
                log.info(f"    {role}: {preview}")

    skipped = 0

    if args.format in ("text", "both"):
        converted_text = []
        for item in alpaca_data:
            result = alpaca_to_chatml_text(item)
            if result:
                converted_text.append(result)
            else:
                skipped += 1

        path_t = out_path("chatml_text")
        with open(path_t, "w", encoding="utf-8") as f:
            json.dump(converted_text, f, ensure_ascii=False, indent=2)

        log.info(f"\n════════════════ text 格式 ════════════════")
        log.info(f"  转换成功: {len(converted_text)} 条")
        if skipped:
            log.warning(f"  跳过(字段缺失): {skipped} 条")
        log.info(f"  输出文件: {path_t}  ({path_t.stat().st_size/1024:.1f} KB)")

        # 样本预览
        log.info("\n  样本预览（第1条）：")
        if converted_text:
            preview = converted_text[0]["text"][:200]
            log.info(f"\n{preview}\n  ...")

    log.info("\n✅ 转换完成")


if __name__ == "__main__":
    main()
