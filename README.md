## 项目概述

这是一个基于 Qwen 大语言模型微调的智能电商客服系统，主要特性包括：
- 使用 LoRA 对大语言模型进行客服领域微调
- RAG（检索增强生成）系统，基于 ChromaDB + BGE-M3 向量模型
- FastAPI 后端，通过 vLLM 实现流式响应
- Gradio Web 演示界面
- 基于 Redis 的会话管理

## 项目结构

```
capstone/
├── train/                    # 模型训练流水线
│   ├── 1.generate_customerservice_data.py   # 通过 LLM API 生成训练数据
│   ├── 2.convert_to_chatml.py               # Alpaca → ChatML 格式转换
│   ├── 3.finetune_qwen_customerservice.py   # LoRA 微调脚本
│   ├── 4.compare_models.py                  # 对比基座模型与微调模型
│   ├── 5.merge_lora_model.py                # 合并 LoRA 权重到基座模型
│   ├── 6.awq_quantize.py                    # AWQ 4-bit 量化
│   ├── 7.compare_quantization.py            # 对比量化前后质量
│   ├── train.json                           # 训练数据（ChatML messages 格式）
│   └── test.json                            # 测试数据
│
├── deploy/                   # 部署与推理
│   ├── app/
│   │   ├── main.py          # FastAPI 入口
│   │   ├── rag.py           # RAG 模块（ChromaDB + BGE-M3）
│   │   ├── session.py       # Redis 会话管理
│   │   ├── query_order.py   # 业务数据库查询层（意图识别）
│   │   ├── mock_db.py       # 演示用模拟数据库
│   │   ├── auth.py          # API Key 鉴权
│   │   └── ratelimit.py     # 限流
│   ├── config.py            # 统一配置（环境变量）
│   ├── gradio_app.py        # Gradio 演示界面
│   └── requirements.txt     # Python 依赖
│
└── models/                   # 本地模型文件
    ├── Qwen1.5-1.8B-Chat/   # 基座模型
    └── bge-m3/              # 向量模型
```

## 训练流水线命令

所有训练脚本需在 `train/` 目录下运行：

```bash
cd train

# 第 1 步：生成训练数据（需要 DEEPSEEK_API_KEY 或 OPENAI_API_KEY）
export DEEPSEEK_API_KEY="sk-..."
python 1.generate_customerservice_data.py --provider deepseek --total 2000

# 第 2 步：将 Alpaca 格式转换为 ChatML messages 格式
python 2.convert_to_chatml.py --input customerservice_alpaca.json --output train.json

# 第 3 步：使用 LoRA 进行微调
python 3.finetune_qwen_customerservice.py

# 第 4 步：对比基座模型与微调模型效果
python 4.compare_models.py

# 第 5 步：将 LoRA adapter 合并到基座模型
python 5.merge_lora_model.py
# 支持命令行参数：
python 5.merge_lora_model.py --base ../models/Qwen1.5-1.8B-Chat --adapter ./lora-adapter --output ./merged-model

# 第 6 步：AWQ 4-bit 量化（用于 vLLM 部署）
python 6.awq_quantize.py
# 支持命令行参数：
python 6.awq_quantize.py --input ./merged-model --output ./awq-model --n-samples 128

# 第 7 步：对比量化前后质量
python 7.compare_quantization.py
```

### 关键训练参数

位于 `3.finetune_qwen_customerservice.py`，可通过环境变量覆盖：
- `MAX_LENGTH = 512` - 最大序列长度（RAG 场景可适当增加）
- `LORA_RANK = 16` - LoRA 秩（16 适合本场景）
- `BATCH_SIZE = 4` - 单卡批次大小（8GB 显存推荐值）
- `GRAD_ACCUM = 4` - 梯度累积步数（等效批次 = 16）
- `EPOCHS = 3` - 训练轮数

## 部署命令

在 `deploy/` 目录下运行：

```bash
cd deploy

# 启动 vLLM 推理服务（量化模型）
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/awq-model \
    --quantization awq \
    --gpu-memory-utilization 0.45 \
    --max-num-seqs 4 \
    --max-model-len 2048 \
    --enable-prefix-caching \
    --served-model-name customerservice \
    --port 8001

# 启动 Redis（会话管理必需）
redis-server

# 启动 FastAPI 后端
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 或直接运行
python -m app.main

# 启动 Gradio 界面
python gradio_app.py
```

### 生产环境部署

```bash
# 生产环境必须设置以下环境变量
export PRODUCTION=true
export API_KEYS="your-secret-key-1,your-secret-key-2"
export CORS_ORIGINS="https://your-domain.com"
```

### 环境变量配置

通过 `deploy/config.py` 配置：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `PRODUCTION` | 空 | 设为 `true` 启用生产模式检查 |
| `API_KEYS` | `dev-key-001` | API 密钥，多个用逗号分隔（生产环境必填） |
| `CORS_ORIGINS` | `*` | 允许的跨域域名，多个用逗号分隔 |
| `VLLM_URL` | `http://localhost:8001/v1/chat/completions` | vLLM 服务地址 |
| `REDIS_URL` | `redis://localhost:6379` | Redis 地址 |
| `RATE_LIMIT` | `20` | 每分钟请求限制 |
| `RATE_BURST` | `5` | 允许的突发额外次数 |
| `HTTP_TIMEOUT` | `60` | HTTP 请求超时秒数 |
| `HEALTH_CHECK_TIMEOUT` | `3` | 健康检查超时秒数 |
| `MODEL_PATH` | `/ai/work/P09/project/merged_model` | 合并/量化后模型路径 |
| `EMBED_MODEL` | `/ai/work/models/bge-m3` | 向量模型路径 |
| `RAG_TOP_K` | `3` | RAG 检索 Top-K |
| `RAG_DISTANCE_THRESHOLD` | `0.6` | RAG 相似度阈值 |
| `PDF_UPLOAD_DIRS` | `/tmp,./uploads` | 允许上传 PDF 的目录 |
| `API_PORT` | `8000` | API 服务端口 |
| `GRADIO_PORT` | `7860` | Gradio 服务端口 |
| `VLLM_PORT` | `8001` | vLLM 服务端口 |

## API 接口

FastAPI 服务提供以下接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/stream` | POST | 流式对话（返回 SSE） |
| `/v1/session/{id}` | DELETE | 清除会话历史 |
| `/health` | GET | 健康检查（vLLM + Redis 状态） |
| `/metrics` | GET | Prometheus 监控指标 |

### 请求格式

```json
{
  "session_id": "uuid或留空",
  "user_id": "user-demo",
  "message": "我的订单什么时候发货？"
}
```

## 系统架构

### 请求处理流程

1. **用户提问** → Gradio/FastAPI 接收消息
2. **意图识别** → `query_order.py` 识别订单/物流/退款/商品等意图
3. **数据库查询** → 从 mock_db（或真实数据库）查询用户订单/退款信息
4. **RAG 兜底** → 若无数据库匹配，`rag.py` 通过 ChromaDB 检索 PDF 知识库
5. **上下文注入** → 检索结果格式化为 `【系统检索结果】` 注入系统提示词
6. **LLM 生成** → vLLM 生成流式响应
7. **会话存储** → Redis 存储对话历史

### RAG 系统

- **向量模型**：BGE-M3（本地路径：`models/bge-m3/`）
- **向量数据库**：ChromaDB，使用余弦相似度
- **分块策略**：400 字符，重叠 50 字符
- **检索参数**：Top-K=3，余弦距离阈值 < 0.6
- **PDF 入库**：通过 `rag.py` 中的 `ingest_pdf()` 函数

### 训练数据格式

训练数据使用 ChatML messages 格式：
```json
{
  "messages": [
    {"role": "system", "content": "你是专业电商客服助手..."},
    {"role": "user", "content": "用户消息或【系统检索结果】+问题"},
    {"role": "assistant", "content": "客服回复"}
  ]
}
```

RAG 样本包含 `【系统检索结果】` 前缀，携带结构化数据（订单状态、物流信息等）。

## 安全特性

### 认证与授权
- API Key 认证：通过 `X-API-Key` 请求头传递
- 生产环境强制要求配置 `API_KEYS`

### 跨域安全
- CORS 配置通过 `CORS_ORIGINS` 环境变量控制
- 生产环境建议配置具体域名，避免使用 `*`

### 文件上传安全
- PDF 上传路径白名单：`PDF_UPLOAD_DIRS` 配置允许的目录
- 强制文件扩展名检查（只允许 `.pdf`）
- 路径遍历攻击防护

### 限流保护
- 滑动窗口算法，每 API Key 独立计数
- Redis 连接失败时降级处理（允许请求通过）

## 并发安全

- **Redis 连接**：使用 `asyncio.Lock` 保护初始化
- **向量模型加载**：使用 `threading.Lock` 保护单例初始化
- **后台任务**：异步保存会话的任务添加异常回调，防止异常静默丢失

## 错误处理

- 全局异常处理器捕获未处理异常
- Redis 操作异常记录日志并降级处理
- JSON 解析异常记录详细信息
- 模型路径不存在时提供友好错误提示

## 依赖安装

训练环境：
```bash
pip install transformers peft datasets accelerate torch
```

部署环境：
```bash
pip install -r deploy/requirements.txt
```

核心依赖包：
- `vllm>=0.4.0` - 推理引擎
- `fastapi>=0.110.0` - Web 框架
- `chromadb>=0.5.0` - 向量数据库
- `sentence-transformers>=2.7.0` - 向量模型
- `redis[asyncio]>=5.0.0` - 会话存储
- `gradio>=4.26.0` - Web 界面
