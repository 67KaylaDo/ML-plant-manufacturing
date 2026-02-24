import streamlit as st
import uuid
from datetime import datetime, timedelta

from ml_plant.storage import init_stores
from ml_plant.synthetic_data import (
    make_assets,
    make_alerts,
    make_telemetry,
    make_maintenance_history,
)
from ml_plant.workflow import run_predictive_maintenance_workflow
from ml_plant.chat_agent import answer_question

# ---------------------------------------------------
# Page Setup
# ---------------------------------------------------
st.set_page_config(
    page_title="ML-plant-manufacturing | Agentic AI MVP",
    layout="wide",
)

st.title("🏭 ML-plant-manufacturing — Agentic AI Multi-Agent MVP")
st.caption(
    "Event-driven, multi-agent predictive maintenance system with Orchestrator, "
    "Signal Triage, Prognostics, and Maintenance Decision agents."
)

# ---------------------------------------------------
# Session State (IMPORTANT: never reset on rerun)
# ---------------------------------------------------
if "stores" not in st.session_state:
    st.session_state.stores = init_stores()

if "last_event" not in st.session_state:
    st.session_state.last_event = None

if "last_run" not in st.session_state:
    st.session_state.last_run = None

if "chat_sidebar" not in st.session_state:
    st.session_state.chat_sidebar = []

if "copilot_messages" not in st.session_state:
    st.session_state.copilot_messages = []

stores = st.session_state.stores

# ---------------------------------------------------
# Sidebar Chat (Optional)
# ---------------------------------------------------
st.sidebar.header("💬 Agentic AI Chat (Grounded)")

def build_sidebar_context():
    return {
        "last_event": st.session_state.last_event,
        "final_packet": (st.session_state.last_run or {}).get("final_packet"),
        "ui_trace": (st.session_state.last_run or {}).get("ui_trace", []),
        "approvals": list(stores["approvals"].values()),
        "work_orders": list(stores["work_orders"].values()),
        "notifications": stores["notifications"][-10:],
        "audit_log": stores["audit"][-10:],
        "alerts_preview": stores.get("alerts", [])[:50],
    }

for msg in st.session_state.chat_sidebar:
    with st.sidebar.chat_message(msg["role"]):
        st.write(msg["content"])

q = st.sidebar.chat_input("Ask about decisions, risks, approvals...")
if q:
    st.session_state.chat_sidebar.append({"role": "user", "content": q})
    with st.sidebar.chat_message("user"):
        st.write(q)

    try:
        reply = answer_question(q, build_sidebar_context())
    except Exception as e:
        reply = f"Error: {e}"

    st.session_state.chat_sidebar.append({"role": "assistant", "content": reply})
    with st.sidebar.chat_message("assistant"):
        st.write(reply)

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1️⃣ Generate Data + Event",
        "2️⃣ Run Workflow",
        "3️⃣ Approval Inbox",
        "4️⃣ Audit + Work Orders",
        "5️⃣ 🤖 AI Copilot",
    ]
)

# ===================================================
# TAB 1 — Generate Synthetic Data + Preview Alerts
# ===================================================
with tab1:
    st.subheader("Synthetic Plant Setup")

    plant_id = st.text_input("Plant ID", value="ML-plant-manufacturing")
    n_assets = st.slider("Number of assets", 4, 20, 8)
    alert_rate = st.slider("Alert volume", 50, 800, 200)
    window_minutes = st.slider("Event window (minutes)", 5, 120, 30)

    # ✅ ONLY RESET WHEN BUTTON CLICKED
    if st.button("Generate / Reset Plant Data", type="primary"):
        st.session_state.stores = init_stores()
        stores = st.session_state.stores  # refresh local reference

        assets = make_assets(plant_id, n_assets)
        for a in assets:
            stores["assets"][a["asset_id"]] = a
            stores["history"][a["asset_id"]] = make_maintenance_history(a["asset_id"])

        # optional: new data -> clear old event/run
        st.session_state.last_event = None
        st.session_state.last_run = None

        st.success(f"Generated {len(assets)} assets + maintenance history.")

    st.divider()

    if st.button("Create SensorAlertEvent"):
        if not stores["assets"]:
            st.warning("Generate plant data first.")
        else:
            event_id = f"evt-{uuid.uuid4().hex[:8]}"
            end = datetime.utcnow().replace(microsecond=0)
            start = end - timedelta(minutes=window_minutes)

            start_time = start.isoformat() + "Z"
            end_time = end.isoformat() + "Z"

            assets = list(stores["assets"].values())
            alerts, _, _ = make_alerts(
                plant_id,
                assets,
                minutes=window_minutes,
                base_rate=alert_rate,
            )
            stores["alerts"] = alerts

            for a in assets:
                key = (a["asset_id"], start_time, end_time)
                stores["telemetry"][key] = make_telemetry(a["asset_id"], start_time, end_time)

            event = {
                "event_type": "SensorAlertEvent",
                "event_id": event_id,
                "plant_id": plant_id,
                "start_time": start_time,
                "end_time": end_time,
            }

            st.session_state.last_event = event
            st.success("Event created.")
            st.json(event)

    # show event & alerts if they exist (persist across tabs)
    if st.session_state.last_event and stores.get("alerts"):
        st.write("### 🔎 Preview: First 10 Alerts")
        st.write(f"Total alerts generated: {len(stores['alerts'])}")
        st.json(stores["alerts"][:10])

        counts = {}
        for a in stores["alerts"]:
            counts[a["asset_id"]] = counts.get(a["asset_id"], 0) + 1

        st.write("### 📊 Alerts per Asset (Top 10)")
        st.json(dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]))

# ===================================================
# TAB 2 — Select Assets + Run Workflow
# ===================================================
with tab2:
    st.subheader("Run Multi-Agent Predictive Maintenance Workflow")

    if not st.session_state.last_event:
        st.info("Create an event first in Tab 1.")
    else:
        st.json(st.session_state.last_event)

        alert_assets = sorted(list({a["asset_id"] for a in stores.get("alerts", [])}))

        st.write("### Select which assets continue into workflow")
        selected_assets = st.multiselect(
            "Assets to include",
            options=alert_assets,
            default=alert_assets[: min(5, len(alert_assets))] if alert_assets else [],
        )

        forced_asset = None
        if len(selected_assets) >= 2:
            forced_asset = selected_assets[1]
            st.info(f"Demo rule: forcing human approval for 2nd selected asset → **{forced_asset}**")
        elif len(selected_assets) == 1:
            st.warning("Select at least 2 assets to guarantee approval in Tab 3.")

        if st.button("Run Workflow Now", type="primary"):
            stores["force_approval_assets"] = [forced_asset] if forced_asset else []
            st.session_state.last_run = run_predictive_maintenance_workflow(
                stores,
                st.session_state.last_event,
                constraints={"selected_asset_ids": selected_assets},
            )

        if st.session_state.last_run:
            result = st.session_state.last_run
            col_left, col_right = st.columns([1.3, 1])

            with col_left:
                st.write("### 🔍 Step-by-step Agent + Tool Trace")
                for step in result["ui_trace"]:
                    with st.expander(step["step"]):
                        st.json(step["output"])

            with col_right:
                st.write("### 📦 Final Maintenance Recommendation Packet")
                st.json(result["final_packet"])

# ===================================================
# TAB 3 — Approval Inbox
# ===================================================
with tab3:
    st.subheader("Approval Inbox (Human-in-the-Loop)")

    pending = [a for a in stores["approvals"].values() if a["status"] == "PENDING"]

    if not pending:
        st.info("No pending approvals.")
    else:
        for appr in pending:
            with st.expander(f"{appr['token']} — {appr['asset_id']} — {appr['recommended_action']}"):
                st.json(appr)

                if st.button(f"Approve {appr['token']}", key=f"approve-{appr['token']}"):
                    stores["approvals"][appr["token"]]["status"] = "APPROVED"
                    st.success("Approved.")

# ===================================================
# TAB 4 — Audit + Work Orders
# ===================================================
with tab4:
    st.subheader("Audit Trail + CMMS Work Orders")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Work Orders")
        st.json(stores["work_orders"])

    with col2:
        st.write("### Audit Log (last 5)")
        st.json(stores["audit"][-5:])

    st.write("### Notifications")
    st.json(stores["notifications"][-10:])

# ===================================================
# TAB 5 — AI Copilot (Grounded Chat)
# ===================================================
with tab5:
    st.subheader("🤖 AI Copilot (Grounded Chat)")

    def build_copilot_context():
        return {
            "last_event": st.session_state.last_event,
            "final_packet": (st.session_state.last_run or {}).get("final_packet"),
            "ui_trace": (st.session_state.last_run or {}).get("ui_trace", []),
            "approvals": list(stores["approvals"].values()),
            "work_orders": list(stores["work_orders"].values()),
            "notifications": stores["notifications"][-20:],
            "audit_log": stores["audit"][-20:],
            "alerts_preview": stores.get("alerts", [])[:50],
        }

    def send_copilot_message(text: str):
        text = (text or "").strip()
        if not text:
            return
        st.session_state.copilot_messages.append({"role": "user", "content": text})

        try:
            reply = answer_question(text, build_copilot_context())
        except Exception as e:
            reply = f"Error: {e}"

        st.session_state.copilot_messages.append({"role": "assistant", "content": reply})
        st.rerun()

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Explain latest recommendation", use_container_width=True):
            send_copilot_message("Explain the latest maintenance recommendation packet and reasoning.")
    with c2:
        if st.button("Show strongest evidence", use_container_width=True):
            send_copilot_message("What is the strongest evidence for the highest risk asset?")
    with c3:
        if st.button("What needs approval?", use_container_width=True):
            send_copilot_message("What needs approval right now? List pending approvals and why.")

    st.divider()

    for msg in st.session_state.copilot_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_text = st.chat_input("Ask the Copilot…")
    if user_text:
        send_copilot_message(user_text)