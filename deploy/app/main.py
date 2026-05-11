# -*- coding: utf-8 -*-
"""
FastAPI 主入口
提供：
  POST   /v1/chat/stream      流式对话
  DELETE /v1/session/{id}     清除会话
  GET    /health              健康检查
  GET    /metrics             Prometheus 指标（可选）
"""
import asyncio
import json
import logging
import sys
import time
import uuid

import httpx
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# 把项目根目录加入 path，确保 import config 能找到
sys.path.insert(0, ".")

from config import (
    VLLM_URL, MODEL_NAME, SYSTEM_PROMPT,
    TEMPERATURE, MAX_TOKENS, API_PORT,
    CORS_ORIGINS, HTTP_TIMEOUT, HEALTH_CHECK_TIMEOUT,
)
from app.auth        import verify_api_key
from app.session     import get_history, save_turn, clear_session, ping as redis_ping
from app.ratelimit   import check_rate_limit
from app.rag         import search, kb_count
from app.query_order import query_order_context

# ─── 日志 ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("api.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─── 后台任务异常处理 ─────────────────────────────────────────────────────────
def _handle_save_task_exception(task: asyncio.Task) -> None:
    """处理异步保存会话任务的异常，防止异常被静默丢弃"""
    try:
        exc = task.exception()
        if exc:
            log.error(f"保存会话后台任务异常: {exc}")
    except asyncio.CancelledError:
        log.debug("保存会话后台任务被取消")
    except Exception as e:
        log.error(f"获取后台任务异常失败: {e}")


# ─── FastAPI 应用 ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="智能电商客服 API",
    version="1.0.0",
    description="基于 Qwen2.5-7B 微调的电商客服系统",
)

# CORS（支持从环境变量配置允许的域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Prometheus 监控指标
Instrumentator().instrument(app).expose(app)


# ─── 请求模型 ─────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str = ""       # 为空时自动生成
    user_id:    str = "user-demo"   # 用户标识（演示默认用 user-demo）
    message:    str


# ─── 核心接口：流式对话 ───────────────────────────────────────────────────────
@app.post("/v1/chat/stream")
async def chat_stream(
    req:     ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    # 自动生成 session_id
    session_id = req.session_id or str(uuid.uuid4())

    # 限流检查
    await check_rate_limit(api_key)

    # 构建消息列表
    history = await get_history(session_id)

    # ── 检索优先级：订单数据库 > PDF 知识库 ──────────────────────────────
    # 1. 先查业务数据库（订单/物流/退款/商品库存）
    context = query_order_context(req.user_id, req.message)
    # 2. 无业务数据时，用 PDF 知识库兜底（商品手册/政策文档）
    if not context:
        context = search(req.message)

    # System Prompt：有检索结果时注入
    system_content = SYSTEM_PROMPT
    if context:
        system_content = f"{SYSTEM_PROMPT}\n\n{context}"

    messages = (
        [{"role": "system", "content": system_content}]
        + history
        + [{"role": "user", "content": req.message}]
    )

    # 请求开始时间（用于延迟日志）
    t_start = time.time()
    first_token_logged = False

    async def generate():
        nonlocal first_token_logged
        full_reply = ""

        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                async with client.stream(
                    "POST", VLLM_URL,
                    json={
                        "model":       MODEL_NAME,
                        "messages":    messages,
                        "stream":      True,
                        "temperature": TEMPERATURE,
                        "max_tokens":  MAX_TOKENS,
                    },
                ) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        log.error(f"vLLM 返回错误 {resp.status_code}: {err}")
                        yield "[ERROR] 推理服务暂时不可用，请稍后重试"
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                            token = (
                                chunk["choices"][0]
                                .get("delta", {})
                                .get("content", "")
                            )
                        except (json.JSONDecodeError, KeyError):
                            continue

                        if token:
                            if not first_token_logged:
                                ttft = (time.time() - t_start) * 1000
                                log.info(
                                    f"session={session_id[:8]}  "
                                    f"TTFT={ttft:.0f}ms  "
                                    f"rag={'yes' if context else 'no'}"
                                )
                                first_token_logged = True
                            full_reply += token
                            yield token

        except httpx.ConnectError:
            log.error("无法连接到 vLLM 服务")
            yield "[ERROR] 推理服务未就绪，请稍后重试"
            return
        except Exception as e:
            log.error(f"流式生成异常: {e}")
            yield "[ERROR] 生成失败，请重试"
            return
        finally:
            # 异步保存会话（不阻塞响应）
            if full_reply:
                task = asyncio.create_task(
                    save_turn(session_id, req.message, full_reply)
                )
                # 添加异常回调，防止后台任务异常被静默丢弃
                task.add_done_callback(_handle_save_task_exception)
                total_ms = (time.time() - t_start) * 1000
                log.info(
                    f"session={session_id[:8]}  "
                    f"total={total_ms:.0f}ms  "
                    f"tokens≈{len(full_reply)}"
                )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "X-Session-Id": session_id,
            "Cache-Control": "no-cache",
        },
    )


# ─── 清除会话 ─────────────────────────────────────────────────────────────────
@app.delete("/v1/session/{session_id}")
async def delete_session(
    session_id: str,
    api_key: str = Depends(verify_api_key),
):
    await clear_session(session_id)
    return {"deleted": session_id}


# ─── 健康检查 ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    # 检查 vLLM
    vllm_ok = False
    try:
        async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
            r = await client.get(
                VLLM_URL.replace("/v1/chat/completions", "/health")
            )
            vllm_ok = r.status_code == 200
    except Exception:
        pass

    # 检查 Redis
    redis_ok = await redis_ping()

    # 知识库状态
    kb_docs = kb_count()

    status = "ok" if (vllm_ok and redis_ok) else "degraded"
    code   = 200 if status == "ok" else 503

    return JSONResponse(
        status_code=code,
        content={
            "status":  status,
            "vllm":    vllm_ok,
            "redis":   redis_ok,
            "kb_docs": kb_docs,
        },
    )


# ─── 全局异常处理 ─────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"未捕获异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ─── 启动入口 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=API_PORT,
        workers=2,
        log_level="info",
    )
