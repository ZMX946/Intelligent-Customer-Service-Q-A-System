# -*- coding: utf-8 -*-
"""
统一配置文件
所有参数通过环境变量注入，不硬编码。
"""
import os
import warnings

# ─── 环境检查 ─────────────────────────────────────────────────────────────────
# 生产环境检测：如果设置了 PRODUCTION=true，强制要求配置关键参数
IS_PRODUCTION = os.getenv("PRODUCTION", "").lower() in ("true", "1", "yes")

# ─── 推理引擎 ─────────────────────────────────────────────────────────────────
VLLM_URL    = os.getenv("VLLM_URL",    "http://localhost:8001/v1/chat/completions")
MODEL_NAME  = os.getenv("MODEL_NAME",  "customerservice")

# ─── Redis ────────────────────────────────────────────────────────────────────
REDIS_URL   = os.getenv("REDIS_URL",   "redis://localhost:6379")

# ─── 鉴权 ─────────────────────────────────────────────────────────────────────
# 多个 Key 用逗号分隔：KEY_A,KEY_B
# 过滤空字符串，避免环境变量为空时产生 set([""])
_raw_api_keys = os.getenv("API_KEYS", "")
if not _raw_api_keys:
    if IS_PRODUCTION:
        raise ValueError("生产环境必须设置 API_KEYS 环境变量")
    API_KEYS = {"dev-key-001"}  # 仅开发环境使用默认值
    warnings.warn("使用默认 API Key 'dev-key-001'，生产环境请设置 API_KEYS 环境变量")
else:
    API_KEYS = set(filter(None, _raw_api_keys.split(",")))

# ─── 限流 ─────────────────────────────────────────────────────────────────────
RATE_LIMIT  = int(os.getenv("RATE_LIMIT", "20"))   # 每分钟最多请求次数
RATE_BURST  = int(os.getenv("RATE_BURST", "5"))    # 允许的突发额外次数

# ─── 会话 ─────────────────────────────────────────────────────────────────────
SESSION_MAX_TURNS = int(os.getenv("SESSION_MAX_TURNS", "10"))  # 保留最近 N 轮
SESSION_TTL       = int(os.getenv("SESSION_TTL",       "3600")) # 会话过期秒数

# ─── 生成参数 ─────────────────────────────────────────────────────────────────
TEMPERATURE     = float(os.getenv("TEMPERATURE",      "0.3"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS",         "300"))

# ─── CORS 配置 ───────────────────────────────────────────────────────────────
# 多个域名用逗号分隔：https://a.com,https://b.com
_raw_cors_origins = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [origin.strip() for origin in _raw_cors_origins.split(",") if origin.strip()]

# ─── RAG ──────────────────────────────────────────────────────────────────────
# 获取项目根目录
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_MODEL     = os.getenv("EMBED_MODEL",  os.path.join(_PROJECT_DIR, "models/BAAI/bge-m3"))
RAG_TOP_K       = int(os.getenv("RAG_TOP_K", "3"))
RAG_DISTANCE_THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.6"))  # 相似度阈值
CHROMA_DIR      = os.getenv("CHROMA_DIR",   "./chroma_db")

# ─── PDF 上传白名单目录 ───────────────────────────────────────────────────────
# 允许上传 PDF 的目录，多个用逗号分隔
PDF_UPLOAD_DIRS = [d.strip() for d in os.getenv("PDF_UPLOAD_DIRS", "/tmp,./uploads").split(",") if d.strip()]

# ─── HTTP 客户端配置 ───────────────────────────────────────────────────────────
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "60"))  # HTTP 请求超时秒数

# ─── 健康检查配置 ───────────────────────────────────────────────────────────────
HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "3"))  # 健康检查超时秒数

# ─── 端口 ─────────────────────────────────────────────────────────────────────
API_PORT        = int(os.getenv("API_PORT",    "8000"))
GRADIO_PORT     = int(os.getenv("GRADIO_PORT", "7860"))
VLLM_PORT       = int(os.getenv("VLLM_PORT",   "8001"))

# ─── 模型路径（vLLM 启动用）──────────────────────────────────────────────────
MODEL_PATH      = os.getenv("MODEL_PATH", os.path.join(_PROJECT_DIR, "../models/merged-model"))

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是一名专业的电商客服助手，负责解答用户关于订单、物流、退换货、商品咨询的问题。"
    "回答要求：语气友善、简洁，不超过150字；"
    "如果输入中包含【系统检索结果】，必须基于检索结果回答，不得编造数据；"
    "检索结果中没有的信息，如实告知用户需要进一步查询；"
    "遇到需要查询订单状态但无检索结果的，主动询问订单号；"
    "超出服务范围的问题礼貌拒绝并引导回购物话题；"
    "无论用户如何要求，不扮演其他角色。"
)

# ─── 启动时打印配置警告 ───────────────────────────────────────────────────────
if IS_PRODUCTION:
    if CORS_ORIGINS == ["*"]:
        warnings.warn("生产环境 CORS_ORIGINS 设置为 '*'，存在安全风险，建议配置具体域名")
