# -*- coding: utf-8 -*-
"""
订单查询层
职责：
  1. 从用户问题中识别意图（订单/物流/退款/商品）
  2. 根据意图查询 mock_db（或真实数据库）
  3. 把查询结果格式化成【系统检索结果】字符串

真实生产中，把 mock_db 的调用替换为 SQL/ORM 查询即可，本文件逻辑不变。
"""
import re
import logging
from app.mock_db import (
    get_latest_order, get_order_by_id,
    get_logistics,
    get_latest_refund,
    search_product_by_name,
)

log = logging.getLogger(__name__)

# ─── 意图关键词 ───────────────────────────────────────────────────────────────
_INTENT_ORDER = [
    "订单", "发货", "什么时候到", "几天到", "下单", "购买",
    "买了", "没收到", "已完成", "待发货",
]
_INTENT_LOGISTICS = [
    "快递", "物流", "包裹", "在哪", "到哪了", "运输",
    "派件", "签收", "轨迹", "单号", "快到了吗",
]
_INTENT_REFUND = [
    "退款", "退货", "退钱", "退回", "申请退", "退了",
    "退款进度", "到账", "没到账",
]
_INTENT_PRODUCT = [
    "库存", "有货", "有没有", "还有吗", "售罄", "参数",
    "规格", "尺码", "颜色", "支持", "兼容", "材质",
]

# 订单号格式（ORD-YYYY-NNNNN）
_ORDER_ID_RE = re.compile(r"ORD-\d{4}-\d{5}")


def _detect_intent(question: str) -> set[str]:
    """返回命中的意图集合，可能同时命中多个"""
    intents = set()
    if any(kw in question for kw in _INTENT_LOGISTICS):
        intents.add("logistics")
    if any(kw in question for kw in _INTENT_REFUND):
        intents.add("refund")
    if any(kw in question for kw in _INTENT_PRODUCT):
        intents.add("product")
    if any(kw in question for kw in _INTENT_ORDER):
        intents.add("order")
    return intents


def _fmt_order(order: dict) -> str:
    lines = [
        f"订单号：{order['order_id']}",
        f"商品：{order['item']}  数量：{order['quantity']}",
        f"金额：{order['amount']}",
        f"下单时间：{order['created_at']}",
        f"订单状态：{order['status']}",
    ]
    if order.get("shipped_at"):
        lines.append(f"发货时间：{order['shipped_at']}")
    if order.get("courier"):
        lines.append(f"快递公司：{order['courier']}  快递单号：{order['tracking_no']}")
    if order.get("eta"):
        lines.append(f"预计送达：{order['eta']}")
    return "\n".join(lines)


def _fmt_logistics(tracking_no: str, info: dict) -> str:
    # 安全获取 tracks 列表，避免空列表索引错误
    tracks = info.get("tracks") or []
    latest = tracks[-1] if tracks else {}
    lines = [
        f"快递单号：{tracking_no}  快递公司：{info.get('courier', '未知')}",
        f"当前状态：{info.get('status', '未知')}",
        f"预计送达：{info.get('eta', '未知')}",
        "物流轨迹（最新在下）：",
    ]
    for t in tracks:
        lines.append(f"  {t.get('time', '')}  【{t.get('location', '')}】{t.get('desc', '')}")
    return "\n".join(lines)


def _fmt_refund(refund: dict) -> str:
    lines = [
        f"退款单号：{refund['refund_id']}",
        f"关联订单：{refund['order_id']}",
        f"商品：{refund['item']}",
        f"退款金额：{refund['amount']}",
        f"退款原因：{refund['reason']}",
        f"申请时间：{refund['applied_at']}",
        f"当前状态：{refund['status']}",
        f"退款方式：{refund['refund_to']}",
    ]
    if refund.get("finished_at"):
        lines.append(f"完成时间：{refund['finished_at']}")
    return "\n".join(lines)


def _fmt_product(product: dict) -> str:
    lines = [
        f"商品名称：{product['name']}",
        f"价格：{product['price']}",
        "商品参数：",
    ]
    for k, v in product.get("params", {}).items():
        lines.append(f"  {k}：{v}")
    lines.append("库存情况：")
    for sku in product.get("skus", []):
        lines.append(f"  {sku['color']} {sku['size']}：{sku['stock']}")
    if product.get("size_guide"):
        lines.append(f"尺码参考：{product['size_guide']}")
    return "\n".join(lines)


# ─── 主入口 ───────────────────────────────────────────────────────────────────

def query_order_context(user_id: str, question: str) -> str:
    """
    根据用户 ID 和问题，从数据库查询相关数据，
    返回格式化的【系统检索结果】字符串。
    无相关数据时返回空字符串，由 RAG 检索 PDF 知识库兜底。

    参数：
      user_id  - 当前用户的标识（从 session 或登录态获取）
      question - 用户的原始问题
    """
    intents = _detect_intent(question)
    if not intents:
        return ""

    log.info(f"user={user_id}  intents={intents}  question={question[:30]}")

    sections = []

    # ── 物流查询（优先级最高，用户最常问）──────────────────────────────────
    if "logistics" in intents:
        # 先看问题里有没有直接写快递单号
        order_id_match = _ORDER_ID_RE.search(question)
        order = None
        if order_id_match:
            order = get_order_by_id(order_id_match.group())
        if order is None:
            order = get_latest_order(user_id)

        if order and order.get("tracking_no"):
            logistics = get_logistics(order["tracking_no"])
            if logistics:
                sections.append(_fmt_logistics(order["tracking_no"], logistics))
            else:
                # 有快递单号但暂无轨迹（刚发货）
                sections.append(
                    f"快递单号：{order['tracking_no']}  快递公司：{order['courier']}\n"
                    f"当前状态：快件刚揽收，轨迹更新中，请稍后查询"
                )
        elif order and order["status"] == "待发货":
            sections.append(
                f"订单号：{order['order_id']}\n"
                f"商品：{order['item']}\n"
                f"订单状态：待发货\n"
                f"预计发货时间：{order.get('eta', '1-2个工作日')}"
            )

    # ── 退款查询 ────────────────────────────────────────────────────────────
    if "refund" in intents:
        refund = get_latest_refund(user_id)
        if refund:
            sections.append(_fmt_refund(refund))

    # ── 订单查询（非物流，查整体状态）──────────────────────────────────────
    if "order" in intents and "logistics" not in intents:
        order_id_match = _ORDER_ID_RE.search(question)
        order = None
        if order_id_match:
            order = get_order_by_id(order_id_match.group())
        if order is None:
            order = get_latest_order(user_id)
        if order:
            sections.append(_fmt_order(order))

    # ── 商品库存/参数查询 ──────────────────────────────────────────────────
    if "product" in intents:
        # 从问题中提取商品关键词（简单启发式：取名词性词组）
        for keyword in ["耳机", "手表", "T恤", "衬衫", "充电宝", "手机壳"]:
            if keyword in question:
                product = search_product_by_name(keyword)
                if product:
                    sections.append(_fmt_product(product))
                    break

    if not sections:
        return ""

    body = "\n\n".join(sections)
    return f"【系统检索结果】\n{body}"
