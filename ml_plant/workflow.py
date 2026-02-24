from __future__ import annotations

import copy
from pydantic import ValidationError

from ml_plant.llm_gemini import GeminiClient
from ml_plant.prompts import ORCHESTRATOR_PROMPT, TRIAGE_PROMPT, PROGNOSTICS_PROMPT, DECISION_PROMPT
from ml_plant.schemas import SensorAlertEvent, SignalTriageOut, PrognosticsOut, MaintenanceDecisionOut, OrchestratorOut
from ml_plant import tools

def _validate(model_cls, obj: dict):
    try:
        return model_cls.model_validate(obj), None
    except ValidationError as e:
        return None, {"error": "INVALID_JSON_SCHEMA", "details": e.errors()}

def _llm(llm: GeminiClient, prompt: str, payload: dict, retries: int = 1) -> dict:
    last = None
    for _ in range(retries + 1):
        try:
            return llm.generate_json(prompt, payload)
        except Exception as e:
            last = str(e)
    return {"error": "LLM_FAILED", "details": last}

def run_predictive_maintenance_workflow(stores: dict, event: dict, constraints: dict | None = None) -> dict:
    llm = GeminiClient()
    ui_trace = []
    constraints = constraints or {"technicians_available": True, "production_blackout_windows": [], "notes": "synthetic constraints"}

    try:
        ev = SensorAlertEvent.model_validate(event)
    except ValidationError as e:
        return {"ui_trace": ui_trace, "final_packet": {"error": "INVALID_EVENT", "details": e.errors()}}

    # T1
    t1 = tools.T1_query_alert_store(stores, ev.plant_id, ev.start_time, ev.end_time)
    ui_trace.append({"step": "T1 QueryAlertStore", "output": t1})

    alerts = t1["alerts"]
    impacted_assets = list({a["asset_id"] for a in alerts})[:5]

    # T4 metadata
    asset_meta = {}
    for aid in impacted_assets:
        try:
            asset_meta[aid] = tools.T4_asset_registry_lookup(stores, aid)
        except Exception:
            continue
    ui_trace.append({"step": "T4 AssetRegistryLookup", "output": asset_meta})

    # A2 triage
    triage_raw = _llm(llm, TRIAGE_PROMPT, {"event_id": ev.event_id, "alerts": alerts, "asset_metadata": asset_meta})
    ui_trace.append({"step": "A2 SignalTriageAgent (raw)", "output": triage_raw})

    triage_obj, triage_err = _validate(SignalTriageOut, triage_raw)
    if triage_err:
        t12 = tools.T12_audit_log_write(stores, ev.event_id, {"phase": "triage", "error": triage_err, "event": event})
        ui_trace.append({"step": "T12 AuditLogWrite", "output": t12})
        return {"ui_trace": ui_trace, "final_packet": {"error": "TRIAGE_INVALID", "details": triage_err}}

    clusters = [c for c in triage_obj.clusters if c.triage_outcome == "send_to_prognostics"][:3]
    if not clusters:
        t12 = tools.T12_audit_log_write(stores, ev.event_id, {"phase": "triage", "note": "no clusters actionable", "triage": triage_raw})
        ui_trace.append({"step": "T12 AuditLogWrite", "output": t12})
        return {"ui_trace": ui_trace, "final_packet": {"error": "NO_ACTIONABLE_CLUSTERS"}}

    prognostics_results = []
    decision_results = []
    execution_results = []

    for cl in clusters:
        aid = cl.asset_id

        # T2 telemetry
        t2 = tools.T2_query_telemetry_store(stores, aid, ev.start_time, ev.end_time)
        ui_trace.append({"step": f"T2 QueryTelemetryStore ({aid})", "output": t2})

        # T3 dq
        t3 = tools.T3_data_quality_check(stores, t2["telemetry_window_id"], t2["telemetry"])
        ui_trace.append({"step": f"T3 DataQualityCheck ({aid})", "output": t3})

        if t2["status"] != "OK" or t3["status"] != "OK":
            tools.T11_notify(stores, "maintenance_lead", f"DEGRADED: insufficient telemetry for {aid}", {"cluster_id": cl.cluster_id})
            continue

        # T5 history
        t5 = tools.T5_maintenance_history_lookup(stores, aid)
        ui_trace.append({"step": f"T5 MaintenanceHistoryLookup ({aid})", "output": t5})

        # T6/T7 models
        t6 = tools.T6_rul_predict(stores, t2["telemetry_window_id"], aid, t3)
        t7 = tools.T7_failure_risk_predict(stores, t2["telemetry_window_id"], aid, t3)
        ui_trace.append({"step": f"T6 RULPredict ({aid})", "output": t6})
        ui_trace.append({"step": f"T7 FailureRiskPredict ({aid})", "output": t7})

        # A3 prognostics
        prog_raw = _llm(llm, PROGNOSTICS_PROMPT, {
            "cluster_id": cl.cluster_id,
            "asset_id": aid,
            "telemetry_window_id": t2["telemetry_window_id"],
            "data_quality_flags": t3["data_quality_flags"],
            "rul_tool_output": t6,
            "risk_tool_output": t7,
            "maintenance_history": t5
        })
        ui_trace.append({"step": f"A3 PrognosticsAgent (raw) ({aid})", "output": prog_raw})

        prog_obj, prog_err = _validate(PrognosticsOut, prog_raw)
        if prog_err:
            continue
        prognostics_results.append(prog_raw)

        # T8 optimizer
        meta = asset_meta.get(aid) or tools.T4_asset_registry_lookup(stores, aid)
        risk_profile = {"rul": prog_obj.rul, "failure_probability": prog_obj.failure_probability, "confidence": prog_obj.confidence_score}
        t8 = tools.T8_maintenance_window_optimize(stores, aid, risk_profile, constraints)
        ui_trace.append({"step": f"T8 MaintenanceWindowOptimize ({aid})", "output": t8})

        # A4 decision
        dec_raw = _llm(llm, DECISION_PROMPT, {
            "asset_metadata": meta,
            "prognostics": prog_raw,
            "constraints": constraints,
            "candidate_windows": t8,
            "policy_summary": {"shutdown_always_requires_approval": True, "planned_maintenance_requires_approval_if_expected_cost_gt": 120000}
        })
        ui_trace.append({"step": f"A4 MaintenanceDecisionAgent (raw) ({aid})", "output": dec_raw})

        dec_obj, dec_err = _validate(MaintenanceDecisionOut, dec_raw)
        if dec_err:
            continue
        decision_results.append(dec_raw)

        # T9 policy
        t9 = tools.T9_policy_check(stores, aid, dec_obj.recommended_action, dec_obj.expected_impact)
        ui_trace.append({"step": f"T9 PolicyCheck ({aid})", "output": t9})

        if t9["requires_human_approval"] and t9["approval_token"]:
            tools.T11_notify(stores, "approver_group", f"Approval required for {aid}: {dec_obj.recommended_action}", {"approval_token": t9["approval_token"], "cluster_id": cl.cluster_id})
            execution_results.append({"asset_id": aid, "status": "PENDING_APPROVAL", "approval_token": t9["approval_token"]})
        else:
            t10 = tools.T10_cmms_create_work_order(stores, aid, dec_obj.recommended_action, dec_obj.time_window, None)
            ui_trace.append({"step": f"T10 CMMSCreateWorkOrder ({aid})", "output": t10})
            execution_results.append({"asset_id": aid, "status": t10["status"], "work_order_id": t10.get("work_order_id")})

    # Audit
    audit_payload = {"event": event, "triage": triage_raw, "prognostics": prognostics_results, "decisions": decision_results, "execution": execution_results, "constraints": constraints}
    t12 = tools.T12_audit_log_write(stores, ev.event_id, audit_payload)
    ui_trace.append({"step": "T12 AuditLogWrite", "output": t12})

    # A1 Orchestrator
    orch_raw = _llm(llm, ORCHESTRATOR_PROMPT, {**copy.deepcopy(audit_payload), "audit_trace_id": t12["audit_trace_id"]})
    ui_trace.append({"step": "A1 OrchestratorAgent (raw)", "output": orch_raw})

    if "audit_trace_id" not in orch_raw and "error" not in orch_raw:
        orch_raw["audit_trace_id"] = t12["audit_trace_id"]

    orch_obj, orch_err = _validate(OrchestratorOut, orch_raw)
    if orch_err:
        return {"ui_trace": ui_trace, "final_packet": {"error": "ORCHESTRATOR_INVALID", "details": orch_err}}

    return {"ui_trace": ui_trace, "final_packet": orch_obj.model_dump()}