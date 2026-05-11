#!/bin/bash
# stop_all.sh — 一键停止所有服务

echo "⏹  停止所有服务..."

# 停止 Gradio
pkill -f "gradio_app.py"    && echo "✅ Gradio 已停止"   || echo "   Gradio 未运行"

# 停止 FastAPI
pkill -f "uvicorn app.main" && echo "✅ FastAPI 已停止"  || echo "   FastAPI 未运行"

# 停止 vLLM
pkill -f "vllm.entrypoints" && echo "✅ vLLM 已停止"    || echo "   vLLM 未运行"

# 停止 Redis
redis-cli shutdown          && echo "✅ Redis 已停止"    || echo "   Redis 未运行"

echo "完成"
