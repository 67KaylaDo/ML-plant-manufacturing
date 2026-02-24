# 🏭 ML-plant-manufacturing — Agentic AI Multi-Agent MVP

This repository contains a Streamlit MVP demo of an **event-driven, multi-agent Agentic AI system** for **predictive maintenance** inspired by the Titan Manufacturing Corporation case study.

The MVP simulates OT/IT integration using synthetic data and demonstrates:
- Alert triage and prioritization
- Prognostics (RUL + failure risk)
- Maintenance decisioning + scheduling
- Policy gating (human-in-the-loop approvals)
- CMMS work order creation (simulated)
- Immutable audit logging
- Grounded AI Copilot chat (Gemini)

---

## ✅ Architecture Overview

### Agents (Multi-Agent System)
1. **A1 Orchestrator Agent**
   - Coordinates the full workflow and produces the final recommendation packet.
   - Enforces policy rules and routes to specialist agents.

2. **A2 Signal Triage Agent**
   - Compresses alert noise into ranked clusters.
   - Prioritizes based on severity, persistence, cross-sensor corroboration, and criticality.

3. **A3 Prognostics Agent**
   - Produces Remaining Useful Life (RUL) + failure probabilities.
   - Uses tool outputs (simulated model endpoints).

4. **A4 Maintenance Decision Agent**
   - Selects one action: `monitor | inspect | planned_maintenance | shutdown_request`
   - Proposes a time window and expected impact.
   - Never executes shutdown autonomously.

### Tools (Passive Executors)
Tools simulate enterprise systems:
- **T1–T5**: Read tools (alerts, telemetry, asset registry, history)
- **T6–T8**: Model/optimizer tools (RUL, risk, scheduling)
- **T9**: Policy check (approval gate)
- **T10–T12**: Write tools (CMMS, Notify, Audit)

### Workflow
Trigger → Context → Triage → Prognostics → Decision → Policy Gate → Execute → Audit → Copilot Explain

---

## 🖥️ UI Walkthrough (Streamlit)
The app provides 5 tabs:
1. **Generate Data + Event**: creates assets, telemetry, alerts, and a SensorAlertEvent trigger
2. **Run Workflow**: runs the multi-agent decision loop and shows step-by-step trace + final packet
3. **Approval Inbox**: simulates human approval for high-impact actions
4. **Audit + Work Orders**: shows work orders, audit log, notifications
5. **AI Copilot**: grounded chat explanations using Gemini

A sidebar chat is also available on every page.

---

## 🚀 Setup & Run (Local)

### 1) Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate