"""
对外服务层：客服调用入口
"""
import json
from datetime import datetime
from pathlib import Path

from .batch import get_current_batch_id
from .excel_io import export_orders_to_excel, import_events_from_excel
from .models import Event, EventType
from .rules import RuleEngine
from .storage import Storage


class FarmOrderService:
    """农场订单数据处理服务"""

    def __init__(self, storage: Storage | None = None, cutoff_hour: int = 9):
        self.storage = storage or Storage()
        self.engine = RuleEngine(self.storage)
        self.cutoff_hour = cutoff_hour

    @property
    def current_batch(self) -> str:
        return get_current_batch_id(self.cutoff_hour)

    # ── Excel 导入 ──

    def import_from_excel(self, file_path: str | Path) -> int:
        """从 Excel 导入事件，自动分配到当前批次，返回导入数量"""
        batch_id = self.current_batch
        events = import_events_from_excel(file_path)

        for ev in events:
            event = Event(
                event_id=None,
                order_id=ev["order_id"],
                event_type=ev["event_type"],
                payload=json.dumps(ev["payload"], ensure_ascii=False),
                batch_id=batch_id,
            )
            self.storage.add_event(event)

        print(f"已导入 {len(events)} 条事件到批次 [{batch_id}]")
        return len(events)

    # ── Excel 导出 ──

    def export_batch_to_excel(self, batch_id: str, output_path: str | Path):
        """将指定批次的订单导出为 Excel"""
        orders = self.storage.get_orders_by_batch(batch_id)
        if not orders:
            print(f"批次 [{batch_id}] 无订单数据")
            return
        export_orders_to_excel(orders, output_path)

    # ── 处理接口 ──

    def process_batch(self, batch_id: str = "") -> int:
        """处理指定批次（默认当前批次）的所有待处理事件"""
        batch_id = batch_id or self.current_batch
        print(f"开始处理批次 [{batch_id}]")
        return self.engine.process_batch(batch_id)

    def register_rule(self, event_type: EventType, handler):
        """注册自定义处理规则"""
        self.engine.register(event_type, handler)

    # ── 手动录入接口（保留，兼容非 Excel 场景）──

    def new_order(self, order_id: str, customer_name: str, address: str,
                  phone: str, items: list[dict]) -> int:
        payload = json.dumps({
            "customer_name": customer_name, "address": address,
            "phone": phone, "items": items,
        }, ensure_ascii=False)
        return self._add_event(order_id, EventType.ORDER_NEW, payload)

    def interrupt_order(self, order_id: str, reason: str = "") -> int:
        payload = json.dumps({"reason": reason}, ensure_ascii=False)
        return self._add_event(order_id, EventType.ORDER_INTERRUPT, payload)

    def resume_order(self, order_id: str, reason: str = "") -> int:
        payload = json.dumps({"reason": reason}, ensure_ascii=False)
        return self._add_event(order_id, EventType.ORDER_RESUME, payload)

    def change_address(self, order_id: str, new_address: str) -> int:
        payload = json.dumps({"new_address": new_address}, ensure_ascii=False)
        return self._add_event(order_id, EventType.AFTER_SALE_ADDRESS, payload)

    def cancel_order(self, order_id: str, reason: str = "") -> int:
        payload = json.dumps({"reason": reason}, ensure_ascii=False)
        return self._add_event(order_id, EventType.AFTER_SALE_CANCEL, payload)

    def _add_event(self, order_id: str, event_type: EventType, payload: str) -> int:
        event = Event(
            event_id=None,
            order_id=order_id,
            event_type=event_type,
            payload=payload,
            batch_id=self.current_batch,
        )
        return self.storage.add_event(event)
