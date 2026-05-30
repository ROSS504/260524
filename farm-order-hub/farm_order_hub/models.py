"""
数据模型定义：订单、事件、批次、规则
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ── 订单状态 ──
class OrderStatus(str, Enum):
    NEW = "new"                # 新增
    ACTIVE = "active"          # 正常
    INTERRUPTED = "interrupted" # 中断
    CANCELLED = "cancelled"    # 已取消
    COMPLETED = "completed"    # 已完成


# ── 事件类型 ──
class EventType(str, Enum):
    # 订单状态类
    ORDER_NEW = "order_new"            # 新增订单
    ORDER_RESUME = "order_resume"      # 中断恢复
    ORDER_INTERRUPT = "order_interrupt" # 正常中断
    # 售后类
    AFTER_SALE_ADDRESS = "after_sale_address"   # 修改地址
    AFTER_SALE_CANCEL = "after_sale_cancel"     # 取消订单


# 事件类型的中文映射，用于 Excel 导入导出
EVENT_TYPE_LABELS = {
    "新增订单": EventType.ORDER_NEW,
    "恢复订单": EventType.ORDER_RESUME,
    "中断订单": EventType.ORDER_INTERRUPT,
    "修改地址": EventType.AFTER_SALE_ADDRESS,
    "取消订单": EventType.AFTER_SALE_CANCEL,
}
EVENT_TYPE_LABELS_REV = {v: k for k, v in EVENT_TYPE_LABELS.items()}

ORDER_STATUS_LABELS = {
    OrderStatus.NEW: "新增",
    OrderStatus.ACTIVE: "正常",
    OrderStatus.INTERRUPTED: "中断",
    OrderStatus.CANCELLED: "已取消",
    OrderStatus.COMPLETED: "已完成",
}


@dataclass
class Order:
    order_id: str
    customer_name: str
    address: str
    phone: str
    items: str              # 商品信息，JSON 字符串
    status: OrderStatus = OrderStatus.NEW
    batch_id: str = ""      # 所属批次
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class Event:
    event_id: int | None    # 自增，新建时为 None
    order_id: str
    event_type: EventType
    payload: str            # JSON 字符串，携带具体变更数据
    batch_id: str = ""      # 所属批次
    processed: bool = False
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Rule:
    """自然语言规则，由用户用自然语言描述，后续交给 Claude Code 解释执行"""
    rule_id: int | None
    content: str            # 自然语言规则描述
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
