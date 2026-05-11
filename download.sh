# 设置 HuggingFace 国内镜像（避免连接 HuggingFace 官方慢或失败）
export HF_ENDPOINT=https://hf-mirror.com

# 安装需要的库
pip install huggingface_hub modelscope


python3 - <<'PY'
# 导入 HuggingFace 下载函数
from huggingface_hub import snapshot_download

# 导入 ModelScope 下载函数（用于下载 Qwen2.5-VL）
from modelscope import snapshot_download as ms_download

import os

# 创建 models 目录（如果不存在）
os.makedirs("./models", exist_ok=True)


# ==============================
# 下载 bge-m3 向量模型
# ==============================
# 该模型用于 RAG 检索（Embedding 模型）
print("开始下载 bge-m3 ...")

snapshot_download(
    repo_id="BAAI/bge-m3",                # HuggingFace 模型名称
    local_dir="./models/bge-m3",          # 本地保存路径
    ignore_patterns=["imgs/*", "*.DS_Store"]  # 忽略无用文件
)


# ==============================
# 下载 Qwen1.5-1.8B-Chat
# ==============================
# 该模型用于文本大模型（客服 / 对话 / 微调）
print("开始下载 Qwen1.5-1.8B-Chat ...")

snapshot_download(
    repo_id="Qwen/Qwen1.5-1.8B-Chat",
    local_dir="./models/Qwen1.5-1.8B-Chat"
)


# 下载完成提示
print("所有模型下载完成！")
PY