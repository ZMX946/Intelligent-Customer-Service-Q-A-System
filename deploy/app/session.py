# -*- coding: utf-8 -*-
"""
Redis 会话管理模块
- 每个会话用 session_id 标识
- 保留最近 SESSION_MAX_TURNS 轮对话
- SESSION_TTL 秒无活动后自动过期
"""
from __future__ import annotations
import json
import logging
import asyncio
from typing import Optional, List, Dict
import redis.asyncio as redis
from config import REDIS_URL, SESSION_MAX_TURNS, SESSION_TTL

log = logging.getLogger(__name__)

# 全局连接池（FastAPI 启动时初始化）
_redis: Optional[redis.Redis] = None
_redis_lock = asyncio.Lock()


async def get_redis() -> redis.Redis:
    """
    获取 Redis 连接。
    使用锁保护初始化，避免多 worker 环境下的竞态条件。
    """
    global _redis
    if _redis is None:
        async with _redis_lock:
            # 双重检查，避免重复初始化
            if _redis is None:
                _redis = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis


def _key(session_id: str) -> str:
    return f"session:{session_id}"


async def get_history(session_id: str) -> List[Dict]:
    """获取会话历史，不存在时返回空列表"""
    r = await get_redis()
    try:
        raw = await r.get(_key(session_id))
        return json.loads(raw) if raw else []
    except json.JSONDecodeError as e:
        log.warning(f"会话数据 JSON 解析失败 (session={session_id}): {e}")
        return []
    except Exception as e:
        log.warning(f"获取会话历史失败 (session={session_id}): {e}")
        return []


async def save_turn(session_id: str, user_msg: str, bot_msg: str) -> None:
    """追加一轮对话，超长时截断最早记录"""
    try:
        r = await get_redis()
        history = await get_history(session_id)

        history.append({"role": "user",      "content": user_msg})
        history.append({"role": "assistant", "content": bot_msg})

        # 超过最大轮数时，删除最早的两条（一问一答为一轮）
        if len(history) > SESSION_MAX_TURNS * 2:
            history = history[-(SESSION_MAX_TURNS * 2):]

        await r.setex(
            _key(session_id),
            SESSION_TTL,
            json.dumps(history, ensure_ascii=False),
        )
    except Exception as e:
        log.error(f"保存会话失败 (session={session_id}): {e}")


async def clear_session(session_id: str) -> None:
    """清除指定会话"""
    try:
        r = await get_redis()
        await r.delete(_key(session_id))
    except Exception as e:
        log.warning(f"清除会话失败 (session={session_id}): {e}")


async def ping() -> bool:
    """检查 Redis 连通性"""
    try:
        r = await get_redis()
        return await r.ping()
    except Exception as e:
        log.warning(f"Redis ping 失败: {e}")
        return False
