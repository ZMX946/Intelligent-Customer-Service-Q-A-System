# 安装需要的库
import subprocess
subprocess.run(["pip", "install", "modelscope"], check=True)

# 导入 ModelScope 下载函数
from modelscope import snapshot_download

import os

# 创建 models 目录（如果不存在）
os.makedirs("./models", exist_ok=True)

# 下载 bge-m3 向量模型
print("开始下载 bge-m3 ...")
snapshot_download(
    model_id="BAAI/bge-m3",                # ModelScope 模型名称
    cache_dir="./models",                  # 本地保存路径
    revision="master"
)

# 下载 Qwen1.5-1.8B-Chat 模型
print("开始下载 Qwen1.5-1.8B-Chat ...")
snapshot_download(
    model_id="Qwen/Qwen1.5-1.8B-Chat",    # ModelScope 模型名称
    cache_dir="./models",                  # 本地保存路径
    revision="master"
)

# 下载完成提示
print("所有模型下载完成！")
