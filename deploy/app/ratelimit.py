# -*- coding: utf-8 -*-
"""
限流模块（滑动窗口算法）
每个 API Key 每分钟最多 RATE_LIMIT 次请求，
允许 RATE_BURST 次突发。
"""
import time
import logging
from fastapi import HTTPException
from app.session import get_redis
from config import RATE_LIMIT, RATE_BURST

log = logging.getLogger(__name__)


async def check_rate_limit(api_key: str) -> None:
    """
    滑动窗口限流。
    用 Redis Sorted Set 记录请求时间戳，
    每次请求时清除 60 秒前的记录，再统计当前窗口内的请求数。
    
    异常处理：
    - Redis 连接失败时降级处理（允许请求通过，但记录警告）
    """
    try:
        r = await get_redis()
    except Exception as e:
        # Redis 连接失败，降级处理：允许请求通过
        log.warning(f"Redis 连接失败，跳过限流检查: {e}")
        return
    
    key = f"rl:{api_key}"
    now = time.time()
    window_start = now - 60.0

    try:
        # 使用 Redis 事务确保原子性
        async with r.pipeline() as pipe:
            # 先检查当前请求数（在 zadd 之前）
            await pipe.zremrangebyscore(key, 0, window_start)
            await pipe.zcard(key)
            results = await pipe.execute()
        
        count = results[1]  # zcard 的结果
        limit = RATE_LIMIT + RATE_BURST

        # 如果已达限制，直接拒绝（不记录本次请求）
        if count >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {RATE_LIMIT} requests/minute. "
                       f"Please retry after a moment."
            )
        
        # 未达限制，记录本次请求
        await r.zadd(key, {f"{now:.6f}": now})
        await r.expire(key, 60)
        
    except HTTPException:
        raise  # 重新抛出限流异常
    except Exception as e:
        # Redis 操作异常，降级处理：允许请求通过
        log.warning(f"Redis 限流操作失败，跳过限流检查: {e}")
        return
