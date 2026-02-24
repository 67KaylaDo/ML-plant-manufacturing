from __future__ import annotations

import copy
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


def run_predictive_maintenance_workflow(
    stores: dict,
    event: dict,
    constraints: dict | None = None,
) -> dict:
    """
    Returns:
      {
        "ui_trace": [ {step, output}, ... ],
        "final_packet": { ... }   # OrchestratorOut or error dict
      }

    Supports:
      constraints["selected_asset_ids"] = list[str]  # filter alerts to these assets
    """
    llm = GeminiClient()
    ui_trace: list[dict] = []

    # ---- constraints defaults + selections ----
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

    # Filter alerts based on selected assets (if provided)
    if selected_asset_ids:
        alerts = [a for a in alerts if a.get("asset_id") in selected_asset_ids]
        ui_trace.append({
            "step": "Filter alerts by selected assets",
            "output": {"selected_asset_ids": sorted(list(selected_asset_ids)), "alerts_remaining": len(alerts)},
        })

    if not alerts:
        # No alerts to process -> audit + stop
        t12 = tools.T12_audit_log_write(stores, ev.event_id, {"event": event, "note": "No alerts after filtering."})
        ui_trace.append({"step": "T12 AuditLogWrite", "output": t12})
        return {"ui_trace": ui_trace, "final_packet": {"error": "NO_ALERTS_TO_PROCESS", "audit_trace_id": t12.get("audit_trace_id")}}

    # Determine impacted assets from remaining alerts
    impacted_assets = list({a.get("asset_id") for a in alerts if a.get("asset_id")})
    impacted_assets = impacted_assets[:10]  # limit for MVP

    # ---- T4: asset registry lookup (for impacted assets) ----
    asset_meta: dict[str, dict] = {}
    for aid in impacted_assets:
        try:
            asset_meta[aid] = tools.T4_asset_registry_lookup(stores, aid)
        except Exception:
            continue
    ui_trace.append({"step": "T4 AssetRegistryLookup", "output": asset_meta})

    # ---- A2: triage agent ----
    triage_payload = {
        "event_id": ev.event_id,
        "alerts": alerts,
        "asset_metadata": asset_meta,
    }
    triage_raw = _llm_call(llm, TRIAGE_PROMPT, triage_payload, retries=1)
    ui_trace.append({"step": "A2 SignalTriageAgent (raw)", "output": triage_raw})

    triage_obj, triage_err = _validate(SignalTriageOut, triage_raw)
    if triage_err:
        t12 = tools.T12_audit_log_write(stores, ev.event_id, {"phase": "triage", "error": triage_err, "triage_raw": triage_raw})
        ui_trace.append({"step": "T12 AuditLogWrite", "output": t12})
        return {"ui_trace": ui_trace, "final_packet": {"error": "TRIAGE_INVALID", "details": triage_err, "audit_trace_id": t12.get("audit_trace_id")}}

    # Choose top clusters for prognostics
    clusters = [c for c in triage_obj.clusters if c.triage_outcome == "send_to_prognostics"]
    clusters = clusters[:3]  # MVP limit

    if not clusters:
        t12 = tools.T12_audit_log_write(stores, ev.event_id, {"phase": "triage", "note": "No actionable clusters", "triage": triage_raw})
        ui_trace.append({"step": "T12 AuditLogWrite", "output": t12})
        return {"ui_trace": ui_trace, "final_packet": {"error": "NO_ACTIONABLE_CLUSTERS", "audit_trace_id": t12.get("audit_trace_id")}}

    prognostics_results: list[dict] = []
    decision_results: list[dict] = []
    execution_results: list[dict] = []

    # ---- per cluster pipeline ----
    for cl in clusters:
        aid = cl.asset_id

        # ---- T2: telemetry ----
        t2 = tools.T2_query_telemetry_store(stores, aid, ev.start_time, ev.end_time)
        ui_trace.append({"step": f"T2 QueryTelemetryStore ({aid})", "output": t2})

        # ---- T3: data quality ----
        t3 = tools.T3_data_quality_check(stores, t2.get("telemetry_window_id"), t2.get("telemetry", []))
        ui_trace.append({"step": f"T3 TelemetryDataQualityCheck ({aid})", "output": t3})

        if t2.get("status") != "OK" or t3.get("status") != "OK":
            tools.T11_notify(stores, "maintenance_lead", f"DEGRADED: insufficient telemetry for {aid}", {"cluster_id": cl.cluster_id})
            execution_results.append({"asset_id": aid, "status": "DEGRADED_INSUFFICIENT_DATA"})
            continue

        # ---- T5: maintenance history ----
        t5 = tools.T5_maintenance_history_lookup(stores, aid)
        ui_trace.append({"step": f"T5 MaintenanceHistoryLookup ({aid})", "output": t5})

        # ---- T6/T7: model tools ----
        t6 = tools.T6_rul_predict(stores, t2.get("telemetry_window_id"), aid, t3)
        t7 = tools.T7_failure_risk_predict(stores, t2.get("telemetry_window_id"), aid, t3)
        ui_trace.append({"step": f"T6 RULPredict ({aid})", "output": t6})
        ui_trace.append({"step": f"T7 FailureRiskPredict ({aid})", "output": t7})

        # ---- A3: prognostics agent ----
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
            execution_results.append({"asset_id": aid, "status": "PROGNOSTICS_INVALID"})
            continue
        prognostics_results.append(prog_raw)

        # ---- T8: window optimizer ----
        risk_profile = {
            "rul": prog_obj.rul,
            "failure_probability": prog_obj.failure_probability,
            "confidence": prog_obj.confidence_score,
        }
        t8 = tools.T8_maintenance_window_optimize(stores, aid, risk_profile, constraints)
        ui_trace.append({"step": f"T8 MaintenanceWindowOptimize ({aid})", "output": t8})

        # ---- A4: maintenance decision agent ----
        meta = asset_meta.get(aid)
        if not meta:
            try:
                meta = tools.T4_asset_registry_lookup(stores, aid)
            except Exception:
                meta = {"asset_id": aid}

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
            execution_results.append({"asset_id": aid, "status": "DECISION_INVALID"})
            continue
        decision_results.append(dec_raw)

        # ---- T9: policy gate ----
        t9 = tools.T9_policy_check(stores, aid, dec_obj.recommended_action, dec_obj.expected_impact)
        ui_trace.append({"step": f"T9 PolicyCheck ({aid})", "output": t9})

        if t9.get("requires_human_approval") and t9.get("approval_token"):
            # notify approver group
            tools.T11_notify(
                stores,
                "approver_group",
                f"Approval required for {aid}: {dec_obj.recommended_action}",
                {"approval_token": t9["approval_token"], "cluster_id": cl.cluster_id},
            )
            execution_results.append({"asset_id": aid, "status": "PENDING_APPROVAL", "approval_token": t9["approval_token"]})
        else:
            # create work order
            t10 = tools.T10_cmms_create_work_order(stores, aid, dec_obj.recommended_action, dec_obj.time_window, None)
            ui_trace.append({"step": f"T10 CMMSCreateWorkOrder ({aid})", "output": t10})
            execution_results.append({"asset_id": aid, "status": t10.get("status"), "work_order_id": t10.get("work_order_id")})

    # ---- T12: audit log ----
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

    # ---- A1: orchestrator consolidates final packet ----
    orch_payload = copy.deepcopy(audit_payload)
    orch_payload["audit_trace_id"] = t12.get("audit_trace_id")

    orch_raw = _llm_call(llm, ORCHESTRATOR_PROMPT, orch_payload, retries=1)
    ui_trace.append({"step": "A1 OrchestratorAgent (raw)", "output": orch_raw})

    # Ensure audit_trace_id exists for schema
    if isinstance(orch_raw, dict) and "audit_trace_id" not in orch_raw and "error" not in orch_raw:
        orch_raw["audit_trace_id"] = t12.get("audit_trace_id")

    orch_obj, orch_err = _validate(OrchestratorOut, orch_raw)
    if orch_err:
        return {"ui_trace": ui_trace, "final_packet": {"error": "ORCHESTRATOR_INVALID", "details": orch_err, "audit_trace_id": t12.get("audit_trace_id")}}

    return {"ui_trace": ui_trace, "final_packet": orch_obj.model_dump()}