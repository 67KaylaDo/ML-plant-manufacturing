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
# Session State Initialization
# ---------------------------------------------------
if "stores" not in st.session_state:
    st.session_state.stores = init_stores()

if "last_event" not in st.session_state:
    st.session_state.last_event = None

if "last_run" not in st.session_state:
    st.session_state.last_run = None

if "chat_sidebar" not in st.session_state:
    st.session_state.chat_sidebar = []

if "chat_copilot" not in st.session_state:
    st.session_state.chat_copilot = []

stores = st.session_state.stores


# ---------------------------------------------------
# Sidebar Agentic Chat
# ---------------------------------------------------
st.sidebar.header("💬 Agentic AI Chat")

ground_context_sidebar = {
    "last_event": st.session_state.last_event,
    "final_packet": st.session_state.last_run["final_packet"]
    if st.session_state.last_run
    else None,
    "approvals": list(stores["approvals"].values()),
    "work_orders": list(stores["work_orders"].values()),
    "notifications": stores["notifications"][-10:],
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
        reply = answer_question(q, ground_context_sidebar)
    except Exception as e:
        reply = f"Error calling Gemini: {e}"

    st.session_state.chat_sidebar.append(
        {"role": "assistant", "content": reply}
    )
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
# TAB 1 — Generate Synthetic Data
# ===================================================
with tab1:
    st.subheader("Synthetic Plant Setup")

    plant_id = st.text_input("Plant ID", value="ML-plant-manufacturing")
    n_assets = st.slider("Number of assets", 4, 20, 8)
    alert_rate = st.slider("Alert volume", 50, 800, 200)
    window_minutes = st.slider("Event window (minutes)", 5, 120, 30)

    if st.button("Generate / Reset Plant Data", type="primary"):
        stores.clear()
        stores.update(init_stores())

        assets = make_assets(plant_id, n_assets)
        for a in assets:
            stores["assets"][a["asset_id"]] = a
            stores["history"][a["asset_id"]] = make_maintenance_history(
                a["asset_id"]
            )

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
                stores["telemetry"][key] = make_telemetry(
                    a["asset_id"], start_time, end_time
                )

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


# ===================================================
# TAB 2 — Run Multi-Agent Workflow
# ===================================================
with tab2:
    st.subheader("Run Multi-Agent Predictive Maintenance Workflow")

    if not st.session_state.last_event:
        st.info("Create an event first in Tab 1.")
    else:
        st.json(st.session_state.last_event)

        if st.button("Run Workflow Now", type="primary"):
            st.session_state.last_run = (
                run_predictive_maintenance_workflow(
                    stores, st.session_state.last_event
                )
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
    st.subheader("Approval Inbox (Policy Gate Simulation)")

    pending = [
        a for a in stores["approvals"].values() if a["status"] == "PENDING"
    ]

    if not pending:
        st.info("No pending approvals.")
    else:
        for appr in pending:
            with st.expander(
                f"{appr['token']} — {appr['asset_id']} — {appr['recommended_action']}"
            ):
                st.json(appr)

                if st.button(
                    f"Approve {appr['token']}",
                    key=f"approve-{appr['token']}",
                ):
                    stores["approvals"][appr["token"]][
                        "status"
                    ] = "APPROVED"
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
# TAB 5 — AI Copilot
# ===================================================
with tab5:
    st.subheader("🤖 AI Copilot (Grounded Chat)")

    ground_context = {
        "last_event": st.session_state.last_event,
        "final_packet": st.session_state.last_run["final_packet"]
        if st.session_state.last_run
        else None,
        "ui_trace": st.session_state.last_run["ui_trace"]
        if st.session_state.last_run
        else None,
        "approvals": list(stores["approvals"].values()),
        "work_orders": list(stores["work_orders"].values()),
        "notifications": stores["notifications"],
    }

    colA, colB, colC = st.columns(3)

    if colA.button("Explain latest recommendation"):
        st.session_state.chat_copilot.append(
            {
                "role": "user",
                "content": "Explain the latest maintenance recommendation packet and reasoning.",
            }
        )

    if colB.button("Show strongest evidence"):
        st.session_state.chat_copilot.append(
            {
                "role": "user",
                "content": "What is the strongest evidence for the highest risk asset?",
            }
        )

    if colC.button("What needs approval?"):
        st.session_state.chat_copilot.append(
            {
                "role": "user",
                "content": "Which actions require human approval right now?",
            }
        )

    for msg in st.session_state.chat_copilot:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    q2 = st.chat_input("Ask the Copilot...")
    if q2:
        st.session_state.chat_copilot.append(
            {"role": "user", "content": q2}
        )

        with st.chat_message("user"):
            st.write(q2)

        try:
            answer = answer_question(q2, ground_context)
        except Exception as e:
            answer = f"Error calling Gemini: {e}"

        st.session_state.chat_copilot.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)