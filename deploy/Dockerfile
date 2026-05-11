# 使用 Python 3.10 作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV VLLM_DEVICE=cpu
ENV VLLM_LOGGING_LEVEL=INFO

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8000 7860 8001

# 启动脚本
CMD ["bash", "start_all.sh"]