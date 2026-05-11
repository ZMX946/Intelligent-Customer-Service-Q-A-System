#!/bin/bash
# start_all.sh — 一键启动所有服务
# 用法：bash start_all.sh

set -e
cd "$(dirname "$0")"
mkdir -p logs

echo "======================================"
echo "  智能电商客服系统 - 启动中"
echo "======================================"

# ── 1. Redis 状态检查 ─────────────────────────────────────────────────────────
echo "⏳ 检查 Redis 连接..."
sleep 3  # 给 Redis 容器启动时间
if redis-cli -h redis ping > /dev/null 2>&1; then
    echo "✅ Redis 连接成功"
else
    echo "❌ Redis 连接失败，请检查 Redis 服务"
    exit 1
fi

# ── 2. vLLM 推理服务 ──────────────────────────────────────────────────────────
VLLM_PORT=${VLLM_PORT:-8001}
MODEL_PATH=${MODEL_PATH:-../train/awq-model}


if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
    echo "✅ vLLM 已在运行（端口 ${VLLM_PORT}）"
else
    echo "⏳ 启动 vLLM（首次加载约需 2 分钟）..."
    python -m vllm.entrypoints.openai.api_server \
        --model          "${MODEL_PATH}" \
        --gpu-memory-utilization 0.45 \
        --max-num-seqs   64 \
        --max-model-len  2048 \
        --enable-prefix-caching \
        --served-model-name customerservice \
        --port           ${VLLM_PORT} \
        --host           0.0.0.0 \
        > logs/vllm.log 2>&1 &

    echo "   等待 vLLM 就绪..."
    for i in $(seq 1 30); do
        sleep 5
        if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
            echo "✅ vLLM 启动成功（${i}×5s）"
            break
        fi
        echo "   已等待 $((i*5))s..."
        if [ $i -eq 30 ]; then
            echo "❌ vLLM 启动超时，请检查 logs/vllm.log"
            exit 1
        fi
    done
fi

# ── 3. FastAPI 服务 ───────────────────────────────────────────────────────────
API_PORT=${API_PORT:-8000}

if curl -s http://localhost:${API_PORT}/health > /dev/null 2>&1; then
    echo "✅ FastAPI 已在运行（端口 ${API_PORT}）"
else
    echo "⏳ 启动 FastAPI..."
    uvicorn app.main:app \
        --host    0.0.0.0 \
        --port    ${API_PORT} \
        --workers 2 \
        --log-level info \
        > logs/api.log 2>&1 &

    echo "   等待 FastAPI 就绪..."
    for i in $(seq 1 10); do
        sleep 2
        if curl -s http://localhost:${API_PORT}/health > /dev/null 2>&1; then
            echo "✅ FastAPI 启动成功（${i}×2s）"
            break
        fi
        if [ $i -eq 10 ]; then
            echo "❌ FastAPI 启动超时，请检查 logs/api.log"
            exit 1
        fi
    done
fi

# ── 4. Gradio 界面 ────────────────────────────────────────────────────────────
GRADIO_PORT=${GRADIO_PORT:-7860}

if curl -s http://localhost:${GRADIO_PORT} > /dev/null 2>&1; then
    echo "✅ Gradio 已在运行（端口 ${GRADIO_PORT}）"
else
    echo "⏳ 启动 Gradio..."
    python gradio_app.py > logs/gradio.log 2>&1 &
    sleep 3
    echo "✅ Gradio 启动成功"
fi

# ── 完成 ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "  🎉 所有服务已就绪"
echo "======================================"
echo "  演示界面:  http://localhost:${GRADIO_PORT}"
echo "  API 文档:  http://localhost:${API_PORT}/docs"
echo "  健康检查:  http://localhost:${API_PORT}/health"
echo ""
echo "  查看日志:"
echo "    tail -f logs/vllm.log"
echo "    tail -f logs/api.log"
echo "    tail -f logs/gradio.log"
echo "======================================"

# 保持容器运行
wait
