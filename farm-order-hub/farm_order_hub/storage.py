"""
SQLite 数据存储层
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "app.db"


class Storage:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            -- 数据类型（如"订单状态变更"、"售后信息"）
            CREATE TABLE IF NOT EXISTS data_types (
                type_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL
            );

            -- 导入的数据集（每次上传一个 Excel = 一条记录）
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                type_id     INTEGER NOT NULL,
                file_name   TEXT NOT NULL,
                row_count   INTEGER NOT NULL DEFAULT 0,
                raw_headers TEXT NOT NULL DEFAULT '[]',
                created_at  TEXT NOT NULL,
                FOREIGN KEY (type_id) REFERENCES data_types(type_id)
            );

            -- 导入的原始数据行
            CREATE TABLE IF NOT EXISTS data_rows (
                row_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id  INTEGER NOT NULL,
                row_data    TEXT NOT NULL,
                processed   INTEGER NOT NULL DEFAULT 0,
                result      TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
            );

            -- 类型绑定的处理规则（自然语言）
            CREATE TABLE IF NOT EXISTS rules (
                rule_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                type_id     INTEGER NOT NULL,
                content     TEXT NOT NULL,
                enabled     INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (type_id) REFERENCES data_types(type_id)
            );

            -- 对话记录（每个类型一个对话窗口）
            CREATE TABLE IF NOT EXISTS chat_messages (
                msg_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                type_id     INTEGER NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                FOREIGN KEY (type_id) REFERENCES data_types(type_id)
            );
        """)
        self.conn.commit()

    # ── 数据类型 ──

    def create_type(self, name: str, description: str = "") -> int:
        now = datetime.now().isoformat()
        cursor = self.conn.execute(
            "INSERT INTO data_types (name, description, created_at) VALUES (?, ?, ?)",
            (name, description, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_type(self, type_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM data_types WHERE type_id = ?", (type_id,)).fetchone()
        return dict(row) if row else None

    def get_type_by_name(self, name: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM data_types WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def get_all_types(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM data_types ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    # ── 数据集 ──

    def create_dataset(self, type_id: int, file_name: str, row_count: int, raw_headers: list[str]) -> int:
        now = datetime.now().isoformat()
        cursor = self.conn.execute(
            "INSERT INTO datasets (type_id, file_name, row_count, raw_headers, created_at) VALUES (?, ?, ?, ?, ?)",
            (type_id, file_name, row_count, json.dumps(raw_headers, ensure_ascii=False), now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_datasets_by_type(self, type_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM datasets WHERE type_id = ? ORDER BY created_at DESC", (type_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_dataset(self, dataset_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        return dict(row) if row else None

    # ── 数据行 ──

    def add_data_rows(self, dataset_id: int, rows_data: list[dict]):
        now = datetime.now().isoformat()
        self.conn.executemany(
            "INSERT INTO data_rows (dataset_id, row_data, created_at) VALUES (?, ?, ?)",
            [(dataset_id, json.dumps(row, ensure_ascii=False), now) for row in rows_data]
        )
        self.conn.commit()

    def get_data_rows(self, dataset_id: int, only_unprocessed: bool = False) -> list[dict]:
        sql = "SELECT * FROM data_rows WHERE dataset_id = ?"
        if only_unprocessed:
            sql += " AND processed = 0"
        sql += " ORDER BY row_id"
        rows = self.conn.execute(sql, (dataset_id,)).fetchall()
        return [dict(r) for r in rows]

    def update_row_result(self, row_id: int, result: str):
        self.conn.execute(
            "UPDATE data_rows SET processed = 1, result = ? WHERE row_id = ?",
            (result, row_id)
        )
        self.conn.commit()

    # ── 规则 ──

    def add_rule(self, type_id: int, content: str) -> int:
        now = datetime.now().isoformat()
        cursor = self.conn.execute(
            "INSERT INTO rules (type_id, content, enabled, created_at) VALUES (?, ?, 1, ?)",
            (type_id, content, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_rules_by_type(self, type_id: int, only_enabled: bool = False) -> list[dict]:
        sql = "SELECT * FROM rules WHERE type_id = ?"
        if only_enabled:
            sql += " AND enabled = 1"
        sql += " ORDER BY created_at"
        rows = self.conn.execute(sql, (type_id,)).fetchall()
        return [dict(r) for r in rows]

    def toggle_rule(self, rule_id: int):
        self.conn.execute(
            "UPDATE rules SET enabled = CASE WHEN enabled=1 THEN 0 ELSE 1 END WHERE rule_id = ?",
            (rule_id,)
        )
        self.conn.commit()

    def delete_rule(self, rule_id: int):
        self.conn.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,))
        self.conn.commit()

    # ── 对话记录 ──

    def add_chat_message(self, type_id: int, role: str, content: str):
        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO chat_messages (type_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (type_id, role, content, now)
        )
        self.conn.commit()

    def get_chat_messages(self, type_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM chat_messages WHERE type_id = ? ORDER BY created_at",
            (type_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def clear_chat(self, type_id: int):
        self.conn.execute("DELETE FROM chat_messages WHERE type_id = ?", (type_id,))
        self.conn.commit()

    def close(self):
        self.conn.close()
