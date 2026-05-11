# -*- coding: utf-8 -*-
"""
模拟业务数据库
模拟电商平台的四张核心表：
  - ORDERS    订单表
  - LOGISTICS 物流轨迹表
  - REFUNDS   退款表
  - PRODUCTS  商品库存表

每个用户都有对应的数据，演示时通过 user_id 查询。
真实生产中替换为数据库连接即可，接口不变。
"""
from datetime import datetime

# ─── 订单表 ──────────────────────────────────────────────────────────────────
# 状态：待付款 / 待发货 / 已发货 / 已完成 / 已取消
ORDERS = {
    "user-001": [
        {
            "order_id":    "ORD-2024-88821",
            "item":        "Nike运动鞋 42码 黑色",
            "quantity":    1,
            "amount":      "¥599.00",
            "status":      "已发货",
            "created_at":  "2024-03-10 14:32",
            "shipped_at":  "2024-03-11 09:15",
            "courier":     "顺丰速运",
            "tracking_no": "SF1234567890",
            "eta":         "2024-03-13",
        },
        {
            "order_id":    "ORD-2024-77001",
            "item":        "运动袜 5双装 白色",
            "quantity":    2,
            "amount":      "¥89.00",
            "status":      "已完成",
            "created_at":  "2024-03-01 10:00",
            "shipped_at":  "2024-03-02 08:30",
            "courier":     "圆通速递",
            "tracking_no": "YT9876543210",
            "eta":         "2024-03-04",
        },
    ],
    "user-002": [
        {
            "order_id":    "ORD-2024-66543",
            "item":        "无线蓝牙耳机 Pro版 黑色",
            "quantity":    1,
            "amount":      "¥328.00",
            "status":      "待发货",
            "created_at":  "2024-03-12 20:18",
            "shipped_at":  None,
            "courier":     None,
            "tracking_no": None,
            "eta":         "预计2024-03-14发货",
        },
    ],
    "user-003": [
        {
            "order_id":    "ORD-2024-55123",
            "item":        "纯棉T恤 M码 白色",
            "quantity":    3,
            "amount":      "¥156.00",
            "status":      "已完成",
            "created_at":  "2024-03-05 09:00",
            "shipped_at":  "2024-03-06 11:00",
            "courier":     "京东物流",
            "tracking_no": "JD1122334455",
            "eta":         "2024-03-08",
        },
    ],
    # 演示用默认账号
    "user-demo": [
        {
            "order_id":    "ORD-2024-99999",
            "item":        "智能手表 黑色 标准版",
            "quantity":    1,
            "amount":      "¥1299.00",
            "status":      "已发货",
            "created_at":  "2024-03-11 16:00",
            "shipped_at":  "2024-03-12 10:30",
            "courier":     "顺丰速运",
            "tracking_no": "SF9988776655",
            "eta":         "2024-03-14",
        },
    ],
}

# ─── 物流轨迹表 ───────────────────────────────────────────────────────────────
# 每个快递单号对应一组时间倒序的轨迹节点
LOGISTICS = {
    "SF1234567890": {
        "courier":  "顺丰速运",
        "status":   "运输中",
        "eta":      "2024-03-13",
        "tracks": [
            {"time": "2024-03-11 09:15", "location": "广州集散中心",   "desc": "快件已发出"},
            {"time": "2024-03-11 18:40", "location": "广州转运中心",   "desc": "快件已离开，下一站上海"},
            {"time": "2024-03-12 06:20", "location": "上海转运中心",   "desc": "快件已到达"},
            {"time": "2024-03-12 14:55", "location": "上海转运中心",   "desc": "快件已离开，下一站杭州"},
        ],
    },
    "YT9876543210": {
        "courier":  "圆通速递",
        "status":   "已签收",
        "eta":      "2024-03-04",
        "tracks": [
            {"time": "2024-03-02 08:30", "location": "北京朝阳营业点", "desc": "快件已揽收"},
            {"time": "2024-03-02 22:10", "location": "北京转运中心",   "desc": "快件已离开"},
            {"time": "2024-03-03 15:30", "location": "杭州转运中心",   "desc": "快件已到达"},
            {"time": "2024-03-04 09:00", "location": "杭州西湖营业点", "desc": "派件中，请保持电话畅通"},
            {"time": "2024-03-04 14:22", "location": "杭州西湖营业点", "desc": "已签收，本人签收"},
        ],
    },
    "JD1122334455": {
        "courier":  "京东物流",
        "status":   "已签收",
        "eta":      "2024-03-08",
        "tracks": [
            {"time": "2024-03-06 11:00", "location": "上海仓库",       "desc": "快件已出库"},
            {"time": "2024-03-07 08:00", "location": "杭州配送站",     "desc": "快件已到达"},
            {"time": "2024-03-08 10:15", "location": "杭州配送站",     "desc": "派件中"},
            {"time": "2024-03-08 15:42", "location": "杭州配送站",     "desc": "已签收，前台代签"},
        ],
    },
    "SF9988776655": {
        "courier":  "顺丰速运",
        "status":   "派件中",
        "eta":      "2024-03-14",
        "tracks": [
            {"time": "2024-03-12 10:30", "location": "深圳集散中心",   "desc": "快件已发出"},
            {"time": "2024-03-12 22:00", "location": "杭州转运中心",   "desc": "快件已到达"},
            {"time": "2024-03-13 08:00", "location": "杭州西湖营业点", "desc": "快件已到达"},
            {"time": "2024-03-14 09:30", "location": "杭州西湖营业点", "desc": "派件中，配送员：张师傅 138xxxx8888"},
        ],
    },
}

# ─── 退款表 ───────────────────────────────────────────────────────────────────
# 状态：审核中 / 退款中 / 退款成功 / 已拒绝
REFUNDS = {
    "user-001": [
        {
            "refund_id":    "REF-2024-11001",
            "order_id":     "ORD-2024-77001",
            "item":         "运动袜 5双装 白色",
            "amount":       "¥89.00",
            "reason":       "商品质量问题",
            "status":       "退款成功",
            "applied_at":   "2024-03-05 10:00",
            "finished_at":  "2024-03-07 14:35",
            "refund_to":    "原路退回至支付宝",
        },
    ],
    "user-003": [
        {
            "refund_id":    "REF-2024-22003",
            "order_id":     "ORD-2024-55123",
            "item":         "纯棉T恤 M码 白色",
            "amount":       "¥156.00",
            "reason":       "尺码不合适",
            "status":       "审核中",
            "applied_at":   "2024-03-10 15:20",
            "finished_at":  None,
            "refund_to":    "原路退回至微信支付",
        },
    ],
    "user-demo": [
        {
            "refund_id":    "REF-2024-99001",
            "order_id":     "ORD-2024-99999",
            "item":         "智能手表 黑色 标准版",
            "amount":       "¥1299.00",
            "reason":       "七天无理由退货",
            "status":       "退款中",
            "applied_at":   "2024-03-13 11:00",
            "finished_at":  None,
            "refund_to":    "原路退回至银行卡",
        },
    ],
}

# ─── 商品库存表 ───────────────────────────────────────────────────────────────
PRODUCTS = {
    "PRD-001": {
        "name":  "无线蓝牙耳机 Pro版",
        "price": "¥328.00",
        "params": {
            "蓝牙版本": "5.3",
            "续航":    "28小时（含充电盒）",
            "防水等级": "IPX5",
            "兼容":    "iOS / Android",
        },
        "skus": [
            {"color": "白色", "size": "均码", "stock": "充足（>100）"},
            {"color": "黑色", "size": "均码", "stock": "紧张（剩余8件）"},
            {"color": "粉色", "size": "均码", "stock": "售罄"},
        ],
    },
    "PRD-002": {
        "name":  "纯棉T恤 基础款",
        "price": "¥52.00",
        "params": {
            "材质":   "100%纯棉",
            "洗涤":   "机洗/手洗均可，水温≤40℃",
            "版型":   "修身",
        },
        "skus": [
            {"color": "白色", "size": "S",  "stock": "充足"},
            {"color": "白色", "size": "M",  "stock": "充足"},
            {"color": "白色", "size": "L",  "stock": "充足"},
            {"color": "白色", "size": "XL", "stock": "售罄"},
            {"color": "黑色", "size": "S",  "stock": "充足"},
            {"color": "黑色", "size": "M",  "stock": "售罄"},
            {"color": "黑色", "size": "L",  "stock": "充足"},
            {"color": "黑色", "size": "XL", "stock": "充足"},
        ],
        "size_guide": "S=155-160cm/85-95斤  M=160-165cm/95-110斤  L=165-170cm/110-125斤  XL=170-175cm/125-140斤",
    },
    "PRD-003": {
        "name":  "智能手表 标准版",
        "price": "¥1299.00",
        "params": {
            "续航":    "7天（普通模式）",
            "防水":    "50米防水",
            "屏幕":    "1.4英寸AMOLED",
            "兼容":    "iOS 12+ / Android 8+",
            "传感器":  "心率/血氧/GPS",
        },
        "skus": [
            {"color": "黑色", "size": "标准版", "stock": "充足"},
            {"color": "银色", "size": "标准版", "stock": "充足"},
            {"color": "金色", "size": "尊享版", "stock": "紧张（剩余3件）"},
        ],
    },
}


# ─── 查询接口（模拟异步数据库查询）──────────────────────────────────────────

def get_orders(user_id: str) -> list[dict]:
    """获取用户所有订单，按下单时间倒序"""
    orders = ORDERS.get(user_id, [])
    return sorted(orders, key=lambda o: o["created_at"], reverse=True)


def get_latest_order(user_id: str) -> dict | None:
    """获取用户最近一笔订单"""
    orders = get_orders(user_id)
    return orders[0] if orders else None


def get_order_by_id(order_id: str) -> dict | None:
    """通过订单号查询订单（不需要 user_id）"""
    for orders in ORDERS.values():
        for o in orders:
            if o["order_id"] == order_id:
                return o
    return None


def get_logistics(tracking_no: str) -> dict | None:
    """查询物流轨迹"""
    return LOGISTICS.get(tracking_no)


def get_refunds(user_id: str) -> list[dict]:
    """获取用户所有退款记录"""
    return REFUNDS.get(user_id, [])


def get_latest_refund(user_id: str) -> dict | None:
    """获取用户最近一笔退款"""
    refunds = get_refunds(user_id)
    return refunds[-1] if refunds else None


def get_product(product_id: str) -> dict | None:
    """查询商品信息"""
    return PRODUCTS.get(product_id)


def search_product_by_name(name: str) -> dict | None:
    """按商品名模糊查询"""
    name = name.lower()
    for pid, p in PRODUCTS.items():
        if name in p["name"].lower():
            return {**p, "product_id": pid}
    return None
