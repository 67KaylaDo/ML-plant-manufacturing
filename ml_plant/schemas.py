from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal

class SensorAlertEvent(BaseModel):
    event_type: Literal["SensorAlertEvent"]
    event_id: str
    plant_id: str
    start_time: str
    end_time: str

class TelemetryWindowReadyEvent(BaseModel):
    event_type: Literal["TelemetryWindowReadyEvent"]
    event_id: str
    plant_id: str
    asset_id: str
    start_time: str
    end_time: str

ReliabilityEvent = SensorAlertEvent | TelemetryWindowReadyEvent

class AlertRecord(BaseModel):
    alert_id: str
    plant_id: str
    asset_id: str
    ts: str
    sensor: str
    code: str
    value: float
    severity: int = Field(ge=1, le=10)

class SignalTriageCluster(BaseModel):
    cluster_id: str
    asset_id: str
    severity_score: float = Field(ge=0, le=100)
    triage_outcome: Literal["send_to_prognostics", "suppress_as_noise", "needs_human_review"]
    reason_codes: list[str]
    data_quality_flags: list[str]

class SignalTriageOut(BaseModel):
    event_id: str
    clusters: list[SignalTriageCluster]

class PrognosticsOut(BaseModel):
    cluster_id: str
    asset_id: str
    telemetry_window_id: str
    model_run_id: str
    rul: dict
    failure_probability: dict
    confidence_score: float = Field(ge=0, le=1)
    data_quality_flags: list[str]
    key_drivers: list[str]

class MaintenanceDecisionOut(BaseModel):
    asset_id: str
    recommended_action: Literal["monitor", "inspect", "planned_maintenance", "shutdown_request"]
    time_window: dict
    expected_impact: dict
    requires_human_approval: bool
    confidence_score: float = Field(ge=0, le=1)
    justification: str
    constraints_considered: list[str]

class RecommendationAsset(BaseModel):
    asset_id: str
    recommended_action: Literal["monitor", "inspect", "planned_maintenance", "shutdown_request"]
    time_window: dict
    requires_human_approval: bool
    confidence_score: float = Field(ge=0, le=1)
    justification: str
    evidence: dict

class OrchestratorOut(BaseModel):
    event_id: str
    plant_id: str
    assets: list[RecommendationAsset]
    overall_status: Literal["OK", "DEGRADED", "ESCALATED", "FAILED"]
    audit_trace_id: str