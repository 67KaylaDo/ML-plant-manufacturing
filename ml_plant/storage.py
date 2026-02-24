from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def append_jsonl(path: str, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path: str, max_lines: int | None = None) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def init_stores() -> dict:
    return {
        "alerts": [],
        "telemetry": {},      # key: (asset_id, start, end) -> list rows
        "assets": {},         # asset_id -> metadata dict
        "history": {},        # asset_id -> history dict
        "work_orders": {},    # work_order_id -> dict
        "notifications": [],  # list
        "audit": [],          # list
        "approvals": {},      # token -> approval dict
    }