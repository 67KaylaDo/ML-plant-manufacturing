from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

SENSORS = ["vibration", "temp", "current", "pressure"]
CODES = ["HI_SPIKE", "SUSTAINED_HIGH", "NOISY_SIGNAL", "DRIFT", "OUT_OF_RANGE"]
ASSET_TYPES = ["CNC", "ROBOT_CELL", "PRESS", "CONVEYOR"]

def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat() + "Z"

def make_assets(plant_id: str, n: int = 8) -> list[dict]:
    assets = []
    for i in range(n):
        asset_type = random.choice(ASSET_TYPES)
        criticality = random.choices(["LOW", "MEDIUM", "HIGH"], weights=[2, 5, 3])[0]
        cost = {"LOW": 20000, "MEDIUM": 80000, "HIGH": 180000}[criticality]
        assets.append({
            "asset_id": f"{asset_type}-{i:03d}",
            "plant_id": plant_id,
            "asset_type": asset_type,
            "criticality": criticality,
            "downtime_cost_per_day": cost,
        })
    return assets

def make_alerts(plant_id: str, assets: list[dict], minutes: int = 30, base_rate: int = 120):
    end = datetime.utcnow()
    start = end - timedelta(minutes=minutes)
    alerts = []
    for _ in range(base_rate):
        a = random.choice(assets)
        ts = start + timedelta(seconds=random.randint(0, minutes * 60))
        sensor = random.choice(SENSORS)
        code = random.choice(CODES)
        severity = random.randint(1, 10)

        if a["criticality"] == "HIGH" and random.random() < 0.35:
            severity = random.randint(7, 10)
            code = random.choice(["HI_SPIKE", "SUSTAINED_HIGH", "OUT_OF_RANGE"])

        alerts.append({
            "alert_id": str(uuid.uuid4())[:8],
            "plant_id": plant_id,
            "asset_id": a["asset_id"],
            "ts": _iso(ts),
            "sensor": sensor,
            "code": code,
            "value": round(random.random() * 100, 3),
            "severity": severity
        })

    for _ in range(int(base_rate * 0.1)):
        if alerts:
            dup = random.choice(alerts).copy()
            dup["alert_id"] = str(uuid.uuid4())[:8]
            alerts.append(dup)

    return alerts, _iso(start), _iso(end)

def make_telemetry(asset_id: str, start_time: str, end_time: str, points: int = 120) -> list[dict]:
    start = datetime.fromisoformat(start_time.replace("Z", ""))
    end = datetime.fromisoformat(end_time.replace("Z", ""))
    total = max(1, int((end - start).total_seconds()))
    rows = []
    for i in range(points):
        t = start + timedelta(seconds=int(i * total / points))
        rows.append({
            "ts": _iso(t),
            "asset_id": asset_id,
            "vibration": round(random.random() * 10, 3),
            "temp": round(40 + random.random() * 40, 3),
            "current": round(10 + random.random() * 25, 3),
            "pressure": round(1 + random.random() * 4, 3)
        })
    return rows

def make_maintenance_history(asset_id: str) -> dict:
    base = datetime.utcnow()
    history = []
    for _ in range(random.randint(2, 6)):
        dt = base - timedelta(days=random.randint(30, 420))
        history.append({
            "ts": _iso(dt),
            "asset_id": asset_id,
            "action": random.choice(["inspect", "replace_bearing", "lubricate", "calibrate"]),
            "notes": random.choice(["routine", "post-alert", "preventive", "corrective"]),
        })
    tickets = []
    for _ in range(random.randint(0, 3)):
        dt = base - timedelta(days=random.randint(1, 90))
        tickets.append({
            "ticket_id": str(uuid.uuid4())[:8],
            "asset_id": asset_id,
            "ts": _iso(dt),
            "summary": random.choice(["noise", "overheat", "vibration high", "quality drift"]),
            "resolved": random.choice([True, False]),
        })
    return {"maintenance_history": history, "recent_tickets": tickets}