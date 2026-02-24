from __future__ import annotations

import copy
from collections import Counter
from typing import Any

from pydantic import ValidationError

from ml_plant.llm_gemini import GeminiClient
from ml_plant.prompts import (
    ORCHESTRATOR_PROMPT,
    TRIAGE_PROMPT,
    PROGNOSTICS_PROMPT,
    DECISION_PROMPT,
)
from ml_plant.schemas import (
    SensorAlertEvent,
    SignalTriageOut,
    PrognosticsOut,
    MaintenanceDecisionOut,
    OrchestratorOut,
)
from ml_plant import tools


def _validate(model_cls, obj: dict) -> tuple[Any | None, dict | None]:
    try:
        return model_cls.model_validate(obj), None
    except ValidationError as e:
        return None, {"error": "INVALID_JSON_SCHEMA", "details": e.errors()}


def _llm_call(llm: GeminiClient, prompt: str, payload: dict, retries: int = 1) -> dict:
    last_err = None
    for _ in range(retries + 1):
        try:
            return llm.generate_json(prompt, payload)
        except Exception as e:
            last_err = str(e)
    return {"error": "LLM_FAILED", "details": last_err}


def _fallback_triage(event_id: str, alerts: list[dict], top_k: int = 3) -> dict:
    """
    Deterministic fallback: choose top assets by alert volume.
    Produces a SignalTriageOut-compatible dict.
    """
    c = Counter([a.get("asset_id") for a in alerts if a.get("asset_id")])
    top = c.most_common(top_k)
    clusters = []
    for i, (asset_id, count) in enumerate(top, start=1):
        clusters.append(
            {
                "cluster_id": f"fb-cluster-{i}",
                "asset_id": asset_id,
                "severity_score": min(100, 30 + count // 3),
                "triage_outcome": "send_to_prognostics",
                "reason_codes": ["FALLBACK_ALERT_VOLUME"],
                "data_quality_flags": [],
            }
        )
    return {"event_id": event_id, "clusters": clusters}


def _fallback_decision(asset_id: str, meta: dict, prog: dict, candidate_windows: dict) -> dict:
    """
    Rule-based fallback decision.
    Uses failure probabilities if present.
    """
    fp = (prog.get("failure_probability") or {})
    p24 = float(fp.get("p24h", 0.0) or 0.0)
    p72 = float(fp.get("p72h", 0.0) or 0.0)

    action = "monitor"
    if p24 >= 0.65:
        action = "shutdown_request"
    elif p72 >= 0.55:
        action = "planned_maintenance"
    elif p72 >= 0.35:
        action = "inspect"

    # pick first candidate window if available
    time_window = {"earliest": None, "latest": None}
    if isinstance(candidate_windows, dict):
        cands = candidate_windows.get("candidate_windows") or []
        if cands:
            time_window = cands[0]

    downtime_cost_per_day = (meta or {}).get("downtime_cost_per_day", 50000)
    expected_cost = 0
    if action in ("planned_maintenance", "shutdown_request"):
        expected_cost = int(downtime_cost_per_day * 0.5)

    return {
        "asset_id": asset_id,
        "recommended_action": action,
        "time_window": time_window,
        "expected_impact": {
            "expected_downtime_cost": expected_cost,
            "risk_reduction_summary": "Fallback decision based on failure probability thresholds.",
        },
        "requires_human_approval": action == "shutdown_request",
        "confidence_score": float(prog.get("confidence_score", 0.5) or 0.5),
        "justification": "Fallback decision used because LLM decision step failed.",
        "constraints_considered": ["FALLBACK_RULES"],
    }


def run_predictive_maintenance_workflow(
    stores: dict,
    event: dict,
    constraints: dict | None = None,
) -> dict:
    llm = GeminiClient()
    ui_trace: list[dict] = []

    constraints = constraints or {}
    constraints.setdefault("technicians_available", True)
    constraints.setdefault("production_blackout_windows", [])
    constraints.setdefault("notes", "synthetic constraints")
    selected_asset_ids = set(constraints.get("selected_asset_ids", []))

    # ---- parse event ----
    try:
        ev = SensorAlertEvent.model_validate(event)
    except ValidationError as e:
        return {"ui_trace": ui_trace, "final_packet": {"error": "INVALID_EVENT", "details": e.errors()}}

    # ---- T1: query alerts ----
    t1 = tools.T1_query_alert_store(stores, ev.plant_id, ev.start_time, ev.end_time)
    ui_trace.append({"step": "T1 QueryAlertStore", "output": t1})
    alerts = t1.get("alerts", [])

    if selected_asset_ids:
        alerts = [a for a in alerts if a.get("asset_id") in selected_asset_ids]
        ui_trace.append({"step": "Filter alerts by selected assets", "output": {"selected_asset_ids": sorted(list(selected_asset_ids)), "alerts_remaining": len(alerts)}})

    if not alerts:
        t12 = tools.T12_audit_log_write(stores, ev.event_id, {"event": event, "note": "No alerts after filtering."})
        ui_trace.append({"step": "T12 AuditLogWrite", "output": t12})
        return {"ui_trace": ui_trace, "final_packet": {"error": "NO_ALERTS_TO_PROCESS", "audit_trace_id": t12.get("audit_trace_id")}}

    impacted_assets = list({a.get("asset_id") for a in alerts if a.get("asset_id")})[:10]

    # ---- T4: asset registry lookup ----
    asset_meta: dict[str, dict] = {}
    for aid in impacted_assets:
        try:
            asset_meta[aid] = tools.T4_asset_registry_lookup(stores, aid)
        except Exception:
            asset_meta[aid] = {"asset_id": aid}
    ui_trace.append({"step": "T4 AssetRegistryLookup", "output": asset_meta})

    # ---- A2 triage (LLM) with fallback ----
    triage_payload = {"event_id": ev.event_id, "alerts": alerts, "asset_metadata": asset_meta}
    triage_raw = _llm_call(llm, TRIAGE_PROMPT, triage_payload, retries=1)
    ui_trace.append({"step": "A2 SignalTriageAgent (raw)", "output": triage_raw})

    triage_obj, triage_err = _validate(SignalTriageOut, triage_raw)
    if triage_err:
        # fallback triage
        triage_raw = _fallback_triage(ev.event_id, alerts, top_k=3)
        ui_trace.append({"step": "A2 SignalTriageAgent (FALLBACK)", "output": triage_raw})
        triage_obj, triage_err = _validate(SignalTriageOut, triage_raw)
        if triage_err:
            t12 = tools.T12_audit_log_write(stores, ev.event_id, {"phase": "triage", "error": triage_err, "triage_raw": triage_raw})
            ui_trace.append({"step": "T12 AuditLogWrite", "output": t12})
            return {"ui_trace": ui_trace, "final_packet": {"error": "TRIAGE_INVALID", "details": triage_err, "audit_trace_id": t12.get("audit_trace_id")}}

    clusters = [c for c in triage_obj.clusters if c.triage_outcome == "send_to_prognostics"][:3]
    if not clusters:
        t12 = tools.T12_audit_log_write(stores, ev.event_id, {"phase": "triage", "note": "No actionable clusters", "triage": triage_raw})
        ui_trace.append({"step": "T12 AuditLogWrite", "output": t12})
        return {"ui_trace": ui_trace, "final_packet": {"error": "NO_ACTIONABLE_CLUSTERS", "audit_trace_id": t12.get("audit_trace_id")}}

    prognostics_results: list[dict] = []
    decision_results: list[dict] = []
    execution_results: list[dict] = []

    for cl in clusters:
        aid = cl.asset_id

        t2 = tools.T2_query_telemetry_store(stores, aid, ev.start_time, ev.end_time)
        ui_trace.append({"step": f"T2 QueryTelemetryStore ({aid})", "output": t2})

        t3 = tools.T3_data_quality_check(stores, t2.get("telemetry_window_id"), t2.get("telemetry", []))
        ui_trace.append({"step": f"T3 TelemetryDataQualityCheck ({aid})", "output": t3})

        if t2.get("status") != "OK" or t3.get("status") != "OK":
            tools.T11_notify(stores, "maintenance_lead", f"DEGRADED: insufficient telemetry for {aid}", {"cluster_id": cl.cluster_id})
            execution_results.append({"asset_id": aid, "status": "DEGRADED_INSUFFICIENT_DATA"})
            continue

        t5 = tools.T5_maintenance_history_lookup(stores, aid)
        ui_trace.append({"step": f"T5 MaintenanceHistoryLookup ({aid})", "output": t5})

        t6 = tools.T6_rul_predict(stores, t2.get("telemetry_window_id"), aid, t3)
        t7 = tools.T7_failure_risk_predict(stores, t2.get("telemetry_window_id"), aid, t3)
        ui_trace.append({"step": f"T6 RULPredict ({aid})", "output": t6})
        ui_trace.append({"step": f"T7 FailureRiskPredict ({aid})", "output": t7})

        prog_payload = {
            "cluster_id": cl.cluster_id,
            "asset_id": aid,
            "telemetry_window_id": t2.get("telemetry_window_id"),
            "data_quality_flags": t3.get("data_quality_flags", []),
            "rul_tool_output": t6,
            "risk_tool_output": t7,
            "maintenance_history": t5,
        }
        prog_raw = _llm_call(llm, PROGNOSTICS_PROMPT, prog_payload, retries=1)
        ui_trace.append({"step": f"A3 PrognosticsAgent (raw) ({aid})", "output": prog_raw})

        prog_obj, prog_err = _validate(PrognosticsOut, prog_raw)
        if prog_err:
            # fallback prognostics using tool outputs directly
            prog_raw = {
                "cluster_id": cl.cluster_id,
                "asset_id": aid,
                "telemetry_window_id": t2.get("telemetry_window_id"),
                "model_run_id": t6.get("model_run_id") or t7.get("model_run_id") or "fb-model",
                "rul": {
                    "estimate_hours": t6.get("rul_estimate", 72),
                    "p10_hours": t6.get("rul_p10", 24),
                    "p90_hours": t6.get("rul_p90", 120),
                },
                "failure_probability": {"p24h": t7.get("p24h", 0.2), "p72h": t7.get("p72h", 0.3), "p7d": t7.get("p7d", 0.4)},
                "confidence_score": float(t6.get("confidence", 0.6)),
                "data_quality_flags": t3.get("data_quality_flags", []),
                "key_drivers": [],
            }
            ui_trace.append({"step": f"A3 PrognosticsAgent (FALLBACK) ({aid})", "output": prog_raw})
            prog_obj, prog_err = _validate(PrognosticsOut, prog_raw)
            if prog_err:
                execution_results.append({"asset_id": aid, "status": "PROGNOSTICS_INVALID"})
                continue

        prognostics_results.append(prog_raw)

        risk_profile = {
            "rul": prog_obj.rul,
            "failure_probability": prog_obj.failure_probability,
            "confidence": prog_obj.confidence_score,
        }
        t8 = tools.T8_maintenance_window_optimize(stores, aid, risk_profile, constraints)
        ui_trace.append({"step": f"T8 MaintenanceWindowOptimize ({aid})", "output": t8})

        meta = asset_meta.get(aid) or {"asset_id": aid}

        dec_payload = {
            "asset_metadata": meta,
            "prognostics": prog_raw,
            "constraints": constraints,
            "candidate_windows": t8,
            "policy_summary": {
                "shutdown_always_requires_approval": True,
                "planned_maintenance_requires_approval_if_expected_cost_gt": 120000,
            },
        }
        dec_raw = _llm_call(llm, DECISION_PROMPT, dec_payload, retries=1)
        ui_trace.append({"step": f"A4 MaintenanceDecisionAgent (raw) ({aid})", "output": dec_raw})

        dec_obj, dec_err = _validate(MaintenanceDecisionOut, dec_raw)
        if dec_err:
            dec_raw = _fallback_decision(aid, meta, prog_raw, t8)
            ui_trace.append({"step": f"A4 MaintenanceDecisionAgent (FALLBACK) ({aid})", "output": dec_raw})
            dec_obj, dec_err = _validate(MaintenanceDecisionOut, dec_raw)
            if dec_err:
                execution_results.append({"asset_id": aid, "status": "DECISION_INVALID"})
                continue

        decision_results.append(dec_raw)

        t9 = tools.T9_policy_check(stores, aid, dec_obj.recommended_action, dec_obj.expected_impact)
        ui_trace.append({"step": f"T9 PolicyCheck ({aid})", "output": t9})

        if t9.get("requires_human_approval") and t9.get("approval_token"):
            tools.T11_notify(
                stores,
                "approver_group",
                f"Approval required for {aid}: {dec_obj.recommended_action}",
                {"approval_token": t9["approval_token"], "cluster_id": cl.cluster_id},
            )
            execution_results.append({"asset_id": aid, "status": "PENDING_APPROVAL", "approval_token": t9["approval_token"]})
        else:
            t10 = tools.T10_cmms_create_work_order(stores, aid, dec_obj.recommended_action, dec_obj.time_window, None)
            ui_trace.append({"step": f"T10 CMMSCreateWorkOrder ({aid})", "output": t10})
            execution_results.append({"asset_id": aid, "status": t10.get("status"), "work_order_id": t10.get("work_order_id")})

    audit_payload = {
        "event": event,
        "constraints": constraints,
        "selected_asset_ids": sorted(list(selected_asset_ids)),
        "triage": triage_raw,
        "prognostics": prognostics_results,
        "decisions": decision_results,
        "execution": execution_results,
    }
    t12 = tools.T12_audit_log_write(stores, ev.event_id, audit_payload)
    ui_trace.append({"step": "T12 AuditLogWrite", "output": t12})

    orch_payload = copy.deepcopy(audit_payload)
    orch_payload["audit_trace_id"] = t12.get("audit_trace_id")

    orch_raw = _llm_call(llm, ORCHESTRATOR_PROMPT, orch_payload, retries=1)
    ui_trace.append({"step": "A1 OrchestratorAgent (raw)", "output": orch_raw})

    if isinstance(orch_raw, dict) and "audit_trace_id" not in orch_raw and "error" not in orch_raw:
        orch_raw["audit_trace_id"] = t12.get("audit_trace_id")

    orch_obj, orch_err = _validate(OrchestratorOut, orch_raw)
    if orch_err:
        # fallback orchestrator packet from decisions
        final_assets = []
        for d in decision_results:
            final_assets.append(
                {
                    "asset_id": d["asset_id"],
                    "recommended_action": d["recommended_action"],
                    "time_window": d["time_window"],
                    "requires_human_approval": bool(tools.T9_policy_check(stores, d["asset_id"], d["recommended_action"], d["expected_impact"]).get("requires_human_approval")),
                    "confidence_score": d.get("confidence_score", 0.5),
                    "justification": d.get("justification", "Fallback orchestrator packet."),
                    "evidence": {"source": "fallback"},
                }
            )
        fallback_packet = {
            "event_id": ev.event_id,
            "plant_id": ev.plant_id,
            "assets": final_assets,
            "overall_status": "DEGRADED",
            "audit_trace_id": t12.get("audit_trace_id"),
        }
        ui_trace.append({"step": "A1 OrchestratorAgent (FALLBACK)", "output": fallback_packet})
        return {"ui_trace": ui_trace, "final_packet": fallback_packet}

    return {"ui_trace": ui_trace, "final_packet": orch_obj.model_dump()}