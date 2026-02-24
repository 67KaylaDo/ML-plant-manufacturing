from __future__ import annotations

import uuid
import random
from ml_plant.storage import now_iso

def T1_query_alert_store(stores: dict, plant_id: str, start_time: str, end_time: str) -> dict:
    alerts = [a for a in stores["alerts"] if a["plant_id"] == plant_id and start_time <= a["ts"] <= end_time]
    telemetry_window_id = f"tw-{uuid.uuid4().hex[:10]}"
    return {"alerts": alerts, "window_metadata": {"telemetry_window_id": telemetry_window_id, "start_time": start_time, "end_time": end_time}}

def T2_query_telemetry_store(stores: dict, asset_id: str, start_time: str, end_time: str) -> dict:
    key = (asset_id, start_time, end_time)
    telemetry = stores["telemetry"].get(key, [])
    status = "OK" if len(telemetry) >= 10 else "INSUFFICIENT_DATA"
    telemetry_window_id = f"tw-{uuid.uuid4().hex[:10]}"
    return {"telemetry": telemetry, "telemetry_window_id": telemetry_window_id, "status": status}

def T3_data_quality_check(stores: dict, telemetry_window_id: str, telemetry: list[dict]) -> dict:
    if len(telemetry) < 10:
        return {"status": "INSUFFICIENT_DATA", "data_quality_flags": ["TOO_FEW_POINTS"]}
    flags = []
    if random.random() < 0.08:
        flags.append("SENSOR_DROPOUT")
    if random.random() < 0.05:
        flags.append("DRIFT_DETECTED")
    return {"status": "OK", "data_quality_flags": flags}

def T4_asset_registry_lookup(stores: dict, asset_id: str) -> dict:
    return stores["assets"][asset_id]

def T5_maintenance_history_lookup(stores: dict, asset_id: str) -> dict:
    return stores["history"][asset_id]

def T6_rul_predict(stores: dict, telemetry_window_id: str, asset_id: str, dq_out: dict) -> dict:
    if dq_out["status"] != "OK":
        return {"status": "INSUFFICIENT_DATA", "model_run_id": f"rul-{uuid.uuid4().hex[:8]}", "rul_estimate": 0, "rul_p10": 0, "rul_p90": 0, "confidence": 0}
    base = random.uniform(12, 120)
    p10 = max(1.0, base * random.uniform(0.5, 0.85))
    p90 = base * random.uniform(1.15, 1.6)
    conf = random.uniform(0.55, 0.92)
    return {"status": "OK", "model_run_id": f"rul-{uuid.uuid4().hex[:8]}", "rul_estimate": round(base, 2), "rul_p10": round(p10, 2), "rul_p90": round(p90, 2), "confidence": round(conf, 2)}

def T7_failure_risk_predict(stores: dict, telemetry_window_id: str, asset_id: str, dq_out: dict) -> dict:
    if dq_out["status"] != "OK":
        return {"status": "INSUFFICIENT_DATA", "model_run_id": f"risk-{uuid.uuid4().hex[:8]}", "p24h": 0, "p72h": 0, "p7d": 0, "confidence": 0}
    p24 = random.uniform(0.02, 0.35)
    p72 = min(0.95, p24 + random.uniform(0.10, 0.35))
    p7d = min(0.99, p72 + random.uniform(0.10, 0.25))
    conf = random.uniform(0.55, 0.92)
    return {"status": "OK", "model_run_id": f"risk-{uuid.uuid4().hex[:8]}", "p24h": round(p24, 2), "p72h": round(p72, 2), "p7d": round(p7d, 2), "confidence": round(conf, 2)}

def T8_maintenance_window_optimize(stores: dict, asset_id: str, risk_profile: dict, constraints: dict) -> dict:
    base_now = now_iso()
    candidate_windows = [{"earliest": base_now, "latest": base_now, "label": f"WindowOption-{i+1}", "notes": "synthetic optimizer output"} for i in range(3)]
    return {"status": "OK", "candidate_windows": candidate_windows}

def T9_policy_check(stores: dict, asset_id: str, recommended_action: str, expected_impact: dict) -> dict:
    requires = False
    if recommended_action == "shutdown_request":
        requires = True
    if recommended_action == "planned_maintenance" and expected_impact.get("expected_downtime_cost", 0) > 120000:
        requires = True

    token = f"appr-{uuid.uuid4().hex[:10]}" if requires else None
    if requires:
        stores["approvals"][token] = {"token": token, "asset_id": asset_id, "recommended_action": recommended_action, "expected_impact": expected_impact, "status": "PENDING", "created_at": now_iso()}
    return {"status": "OK", "requires_human_approval": requires, "approval_token": token}

def T10_cmms_create_work_order(stores: dict, asset_id: str, action: str, time_window: dict, approval_token: str | None) -> dict:
    if approval_token:
        appr = stores["approvals"].get(approval_token)
        if not appr or appr["status"] != "APPROVED":
            return {"status": "FAILED", "work_order_id": None}

    wo_id = f"WO-{uuid.uuid4().hex[:8]}"
    stores["work_orders"][wo_id] = {"work_order_id": wo_id, "asset_id": asset_id, "action": action, "time_window": time_window, "created_at": now_iso()}
    return {"status": "OK", "work_order_id": wo_id}

def T11_notify(stores: dict, recipient_group: str, message: str, links: dict | None = None) -> dict:
    stores["notifications"].append({"ts": now_iso(), "recipient_group": recipient_group, "message": message, "links": links or {}})
    return {"status": "OK"}

def T12_audit_log_write(stores: dict, event_id: str, payload: dict) -> dict:
    audit_id = f"audit-{uuid.uuid4().hex[:10]}"
    stores["audit"].append({"audit_trace_id": audit_id, "event_id": event_id, "ts": now_iso(), "payload": payload})
    return {"status": "OK", "audit_trace_id": audit_id}