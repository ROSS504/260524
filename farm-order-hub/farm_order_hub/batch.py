"""
批次管理：根据截止时间（默认每日09:00）自动划分批次

- 当前时间 < 今日09:00 → 属于今日批次（如 "2026-05-22"）
- 当前时间 >= 今日09:00 → 属于明日批次（如 "2026-05-23"）
"""
from datetime import date, datetime, time, timedelta

DEFAULT_CUTOFF_HOUR = 9  # 截止时间：09:00


def get_current_batch_id(cutoff_hour: int = DEFAULT_CUTOFF_HOUR) -> str:
    """根据当前时间和截止小时，返回当前所属批次 ID（日期字符串）"""
    now = datetime.now()
    cutoff_today = datetime.combine(now.date(), time(cutoff_hour, 0))

    if now < cutoff_today:
        # 九点之前，属于今日批次
        return now.date().isoformat()
    else:
        # 九点及之后，归入明日批次
        return (now.date() + timedelta(days=1)).isoformat()
