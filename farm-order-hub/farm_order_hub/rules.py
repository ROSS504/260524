"""
规则引擎：根据事件类型分派处理逻辑

用法:
    engine = RuleEngine(storage)
    engine.register(EventType.ORDER_NEW, my_handler)   # 注册自定义规则
    engine.process_all()                                # 处理所有未处理事件
"""
import json
from collections import defaultdict
from typing import Callable

from .models import Event, EventType, OrderStatus
from .storage import Storage

# 规则处理函数签名: (storage, event) -> None
RuleHandler = Callable[[Storage, Event], None]


class RuleEngine:
    def __init__(self, storage: Storage):
        self.storage = storage
        self._handlers: dict[EventType, list[RuleHandler]] = defaultdict(list)
        self._register_defaults()

    def register(self, event_type: EventType, handler: RuleHandler):
        """注册一个事件处理规则，同一事件类型可注册多个handler"""
        self._handlers[event_type].append(handler)

    def process_all(self) -> int:
        """处理所有未处理的事件，返回处理数量"""
        return self._process_events(self.storage.get_unprocessed_events())

    def process_batch(self, batch_id: str) -> int:
        """处理指定批次的未处理事件"""
        return self._process_events(self.storage.get_unprocessed_events(batch_id))

    def _process_events(self, events: list) -> int:
        count = 0
        for event in events:
            self._dispatch(event)
            self.storage.mark_event_processed(event.event_id)
            count += 1
        return count

    def _dispatch(self, event: Event):
        handlers = self._handlers.get(event.event_type, [])
        if not handlers:
            print(f"[WARN] 事件类型 {event.event_type.value} 无注册handler，跳过")
            return
        for handler in handlers:
            handler(self.storage, event)

    # ── 内置默认规则 ──

    def _register_defaults(self):
        self.register(EventType.ORDER_NEW, _handle_order_new)
        self.register(EventType.ORDER_RESUME, _handle_order_resume)
        self.register(EventType.ORDER_INTERRUPT, _handle_order_interrupt)
        self.register(EventType.AFTER_SALE_ADDRESS, _handle_address_change)
        self.register(EventType.AFTER_SALE_CANCEL, _handle_cancel)


# ── 默认处理函数 ──

def _handle_order_new(storage: Storage, event: Event):
    """新增订单：从 payload 创建订单记录"""
    data = json.loads(event.payload)
    from .models import Order
    order = Order(
        order_id=event.order_id,
        customer_name=data.get("customer_name", ""),
        address=data.get("address", ""),
        phone=data.get("phone", ""),
        items=json.dumps(data.get("items", []), ensure_ascii=False),
        status=OrderStatus.NEW,
        batch_id=event.batch_id,
    )
    storage.save_order(order)
    print(f"[NEW] 订单 {event.order_id} 已创建")


def _handle_order_resume(storage: Storage, event: Event):
    """中断恢复：将订单状态从 interrupted -> active"""
    order = storage.get_order(event.order_id)
    if order and order.status == OrderStatus.INTERRUPTED:
        storage.update_order_status(event.order_id, OrderStatus.ACTIVE)
        print(f"[RESUME] 订单 {event.order_id} 已恢复")
    else:
        print(f"[SKIP] 订单 {event.order_id} 状态非中断，无法恢复")


def _handle_order_interrupt(storage: Storage, event: Event):
    """正常中断：将订单状态 -> interrupted"""
    storage.update_order_status(event.order_id, OrderStatus.INTERRUPTED)
    print(f"[INTERRUPT] 订单 {event.order_id} 已中断")


def _handle_address_change(storage: Storage, event: Event):
    """修改地址"""
    data = json.loads(event.payload)
    new_address = data.get("new_address", "")
    if new_address:
        storage.update_order_address(event.order_id, new_address)
        print(f"[ADDRESS] 订单 {event.order_id} 地址已更新")


def _handle_cancel(storage: Storage, event: Event):
    """取消订单"""
    storage.update_order_status(event.order_id, OrderStatus.CANCELLED)
    print(f"[CANCEL] 订单 {event.order_id} 已取消")
