# -*- coding: utf-8 -*-
"""
Gradio 演示界面
功能：
  - 左侧：上传产品手册 PDF，查看知识库状态
  - 右侧：多轮对话界面，支持流式输出
  - 对话自动携带 RAG 检索结果
"""
import sys
import uuid
import logging
import httpx
import gradio as gr

sys.path.insert(0, ".")
from config import API_PORT, API_KEYS, GRADIO_PORT
from app.rag import ingest_pdf, kb_count, clear_knowledge_base

log = logging.getLogger(__name__)

# ─── 配置 ─────────────────────────────────────────────────────────────────────
API_BASE = f"http://fastapi:{API_PORT}"
API_KEY  = next(iter(API_KEYS))          # 取第一个 key 用于界面请求
HEADERS  = {"X-API-Key": API_KEY}

# 每个浏览器会话用独立 session_id（Gradio State）


# ─── PDF 上传处理 ─────────────────────────────────────────────────────────────
def handle_upload(file) -> str:
    if file is None:
        return "⚠️ 请选择文件"
    try:
        count = ingest_pdf(file.name)
        total = kb_count()
        return f"✅ 已入库 {count} 个文本块，知识库共 {total} 块"
    except PermissionError as e:
        return f"❌ 安全限制：{e}"
    except ValueError as e:
        return f"❌ 文件格式错误：{e}"
    except Exception as e:
        log.exception("PDF 上传失败")
        return f"❌ 上传失败：{e}"


def handle_clear_kb() -> str:
    try:
        clear_knowledge_base()
        return "🗑️ 知识库已清空"
    except Exception as e:
        return f"❌ 清空失败：{e}"


def get_kb_status() -> str:
    count = kb_count()
    if count == 0:
        return "📭 知识库为空"
    return f"📚 知识库共 {count} 个文本块"


# ─── 对话处理 ─────────────────────────────────────────────────────────────────
def chat(message: str, history: list, session_id: str, user_id: str):
    """
    流式对话生成器。
    history 是 Gradio ChatInterface 格式：[{"role": ..., "content": ...}, ...]
    """
    if not message.strip():
        yield history
        return

    # 追加用户消息
    history = history + [{"role": "user", "content": message}]
    yield history   # 立即显示用户消息

    # 调用 API 流式获取回答
    reply = ""
    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/v1/chat/stream",
            json={"session_id": session_id, "user_id": user_id, "message": message},
            headers=HEADERS,
            timeout=60,
        ) as resp:
            if resp.status_code == 429:
                history = history + [{"role": "assistant", "content": "⚠️ 请求频率超限，请稍后再试"}]
                yield history
                return
            if resp.status_code == 401:
                history = history + [{"role": "assistant", "content": "⚠️ API Key 无效"}]
                yield history
                return
            if resp.status_code != 200:
                history = history + [{"role": "assistant", "content": f"⚠️ 服务异常（{resp.status_code}）"}]
                yield history
                return

            # 逐 token 追加，触发流式更新
            history = history + [{"role": "assistant", "content": ""}]
            for token in resp.iter_text():
                reply += token
                history[-1] = {"role": "assistant", "content": reply}
                yield history

    except httpx.ConnectError:
        history = history + [{"role": "assistant", "content": "⚠️ 无法连接到客服服务，请确认服务已启动"}]
        yield history
    except Exception as e:
        history = history + [{"role": "assistant", "content": f"⚠️ 发生错误：{e}"}]
        yield history


def clear_chat(session_id: str):
    """清空对话记录，同时通知 API 清除服务端会话"""
    try:
        httpx.delete(
            f"{API_BASE}/v1/session/{session_id}",
            headers=HEADERS,
            timeout=5,
        )
    except Exception:
        pass
    # 重新生成 session_id
    new_sid = str(uuid.uuid4())
    return [], new_sid


def fmt_sid(sid: str) -> str:
    """将 session_id 截断显示为前8位"""
    return str(sid)[:8] + "..."


# ─── 界面构建 ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="智能电商客服助手",
) as demo:

    gr.Markdown("## 🛒 智能电商客服助手")
    gr.Markdown("基于 Qwen1.5-1.8B-Chat 微调模型 · 支持 RAG 知识检索 · 流式输出")

    # 每个用户的会话 ID（存在 Gradio State 中）
    # ✅ 修复：value 直接赋字符串，不用 lambda，避免 State 把函数本身传下去
    session_id = gr.State(value=str(uuid.uuid4()))

    with gr.Row():

        # ── 左侧：知识库管理 ────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📚 产品知识库")

            kb_status = gr.Textbox(
                label="知识库状态",
                value=get_kb_status,
                interactive=False,
                every=10,   # 每 10 秒自动刷新
            )

            pdf_input = gr.File(
                label="上传产品手册（PDF）",
                file_types=[".pdf"],
            )
            upload_status = gr.Textbox(
                label="上传状态",
                interactive=False,
            )
            pdf_input.change(
                fn=handle_upload,
                inputs=pdf_input,
                outputs=upload_status,
            )

            clear_kb_btn = gr.Button("🗑️ 清空知识库", variant="secondary")
            clear_kb_btn.click(
                fn=handle_clear_kb,
                outputs=upload_status,
            )

            gr.Markdown("---")
            gr.Markdown(
                "**使用说明**\n\n"
                "1. 上传产品手册 PDF，系统自动建立知识库\n"
                "2. 在右侧输入问题，系统会自动检索相关内容\n"
                "3. 点击「清空对话」开始新会话\n\n"
                "**支持问题类型**\n\n"
                "- 📦 订单与物流查询\n"
                "- 🔄 退款退货申请\n"
                "- 🛍️ 商品规格咨询\n"
                "- 🆘 售后投诉处理"
            )

        # ── 右侧：对话区 ──────────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 💬 在线客服")

            with gr.Row():
                with gr.Column(scale=2):
                    user_id_input = gr.Dropdown(
                        label="演示用户",
                        choices=["user-demo", "user-001", "user-002", "user-003"],
                        value="user-demo",
                        info="选择不同用户可查看不同的订单/物流/退款数据",
                    )
                with gr.Column(scale=3):
                    gr.Markdown(
                        "**用户数据说明**\n"
                        "- user-demo：智能手表已发货，有退款中\n"
                        "- user-001：Nike鞋已发货，运动袜已完成退款\n"
                        "- user-002：耳机待发货\n"
                        "- user-003：T恤退款审核中",
                    )

            chatbot = gr.Chatbot(
                label="对话记录",
                height=500,
                avatar_images=(
                    None,
                    "https://img.icons8.com/color/48/bot.png",
                ),
                render_markdown=True,
                autoscroll=True,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="请输入您的问题，例如：我的订单什么时候发货？",
                    label="",
                    scale=4,
                    lines=2,
                    max_lines=4,
                    autofocus=True,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
                # ✅ 修复：初始值改为空字符串，由 demo.load 统一填充
                sid_display = gr.Textbox(
                    label="会话 ID",
                    interactive=False,
                    scale=3,
                    value="",
                )

            # 发送消息
            send_btn.click(
                fn=chat,
                inputs=[msg_input, chatbot, session_id, user_id_input],
                outputs=chatbot,
            ).then(
                fn=lambda: "",       # 清空输入框
                outputs=msg_input,
            )

            # 回车发送
            msg_input.submit(
                fn=chat,
                inputs=[msg_input, chatbot, session_id, user_id_input],
                outputs=chatbot,
            ).then(
                fn=lambda: "",
                outputs=msg_input,
            )

            # 清空对话
            clear_btn.click(
                fn=clear_chat,
                inputs=session_id,
                outputs=[chatbot, session_id],
            ).then(
                # ✅ 修复：用具名函数 fmt_sid，避免 lambda 接收到函数对象
                fn=fmt_sid,
                inputs=session_id,
                outputs=sid_display,
            )

            # 初始化 session_id 显示
            demo.load(
                # ✅ 修复：同上，用具名函数
                fn=fmt_sid,
                inputs=session_id,
                outputs=sid_display,
            )


# ─── 启动 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        show_error=True,
        theme = gr.themes.Soft().set(
             body_background_fill="#e5e7eb"
        ),
        css="""
        body {
            background-color: #808080;
        }
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        """
)
