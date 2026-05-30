"""
完整使用示例：
1. 生成客服录入 Excel 模板
2. 从 Excel 导入事件（自动按九点截止分批）
3. 处理当前批次
4. 导出处理结果 Excel 给下一环节
"""
from pathlib import Path

from farm_order_hub.batch import get_current_batch_id
from farm_order_hub.excel_io import generate_import_template
from farm_order_hub.service import FarmOrderService
from farm_order_hub.storage import Storage

# 每次示例用全新数据库
db_path = Path("data/demo.db")
if db_path.exists():
    db_path.unlink()

storage = Storage(db_path)
svc = FarmOrderService(storage=storage, cutoff_hour=9)

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# ── Step 1: 生成客服录入模板 ──
template_path = output_dir / "客服录入模板.xlsx"
generate_import_template(template_path)

# ── Step 2: 从 Excel 导入（这里直接用模板里的示例数据）──
batch_id = svc.current_batch
print(f"\n当前批次: {batch_id}")
print("=" * 50)

count = svc.import_from_excel(template_path)

# ── Step 3: 处理当前批次 ──
print("\n" + "=" * 50)
processed = svc.process_batch(batch_id)
print(f"\n共处理 {processed} 条事件")

# ── Step 4: 导出给下一环节 ──
export_path = output_dir / f"发货单_{batch_id}.xlsx"
svc.export_batch_to_excel(batch_id, export_path)

# 打印最终订单状态
print("\n── 订单最终状态 ──")
for order in storage.get_orders_by_batch(batch_id):
    print(f"  {order.order_id}: {order.customer_name} | 状态={order.status.value} | 地址={order.address}")

storage.close()
