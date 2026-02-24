import streamlit as st
import uuid
from datetime import datetime, timedelta

from ml_plant.storage import init_stores
from ml_plant.synthetic_data import make_assets, make_alerts, make_telemetry, make_maintenance_history
from ml_plant.workflow import run_predictive_maintenance_workflow

st.set_page_config(page_title="ML-plant-manufacturing | Agentic AI MVP", layout="wide")
st.title("ML-plant-manufacturing — Agentic AI MVP Demo (Multi-Agent Predictive Maintenance)")

if "stores" not in st.session_state:
    st.session_state.stores = init_stores()
if "last_event" not in st.session_state:
    st.session_state.last_event = None
if "last_run" not in st.session_state:
    st.session_state.last_run = None

stores = st.session_state.stores

tab1, tab2, tab3, tab4 = st.tabs(["1) Generate Data + Event", "2) Run Workflow", "3) Approval Inbox", "4) Audit + Work Orders"])

with tab1:
    st.subheader("Synthetic plant + events")
    plant_id = st.text_input("Plant ID", value="ML-plant-manufacturing")
    n_assets = st.slider("Number of assets", 4, 20, 8)
    alert_rate = st.slider("Alert volume (approx)", 50, 800, 180)
    window_minutes = st.slider("Event window (minutes)", 5, 120, 30)

    colA, colB = st.columns(2)
    with colA:
        if st.button("Generate/Reset Plant Data", type="primary"):
            stores.clear()
            stores.update(init_stores())

            assets = make_assets(plant_id, n_assets)
            for a in assets:
                stores["assets"][a["asset_id"]] = a
                stores["history"][a["asset_id"]] = make_maintenance_history(a["asset_id"])
            st.success(f"Generated {len(assets)} assets + maintenance history.")
    with colB:
        st.write("Current assets (sample):")
        st.json(list(stores["assets"].values())[: min(6, len(stores["assets"]))])

    st.divider()

    if st.button("Create SensorAlertEvent (generate alerts+telemetry)", type="secondary"):
        if not stores["assets"]:
            st.warning("Generate plant data first.")
        else:
            event_id = f"evt-{uuid.uuid4().hex[:8]}"
            end = datetime.utcnow().replace(microsecond=0)
            start = end - timedelta(minutes=window_minutes)
            start_time = start.isoformat() + "Z"
            end_time = end.isoformat() + "Z"

            assets = list(stores["assets"].values())
            alerts, _, _ = make_alerts(plant_id, assets, minutes=window_minutes, base_rate=alert_rate)
            stores["alerts"] = alerts

            for a in assets:
                key = (a["asset_id"], start_time, end_time)
                stores["telemetry"][key] = make_telemetry(a["asset_id"], start_time, end_time, points=120)

            event = {"event_type": "SensorAlertEvent", "event_id": event_id, "plant_id": plant_id, "start_time": start_time, "end_time": end_time}
            st.session_state.last_event = event
            st.success("Event created and stores populated.")
            st.json(event)

with tab2:
    st.subheader("Run the multi-agent workflow (A2 → A3 → A4 → A1)")
    if not st.session_state.last_event:
        st.info("Go to Tab 1 and create a SensorAlertEvent first.")
    else:
        st.json(st.session_state.last_event)
        if st.button("Run Workflow Now", type="primary"):
            st.session_state.last_run = run_predictive_maintenance_workflow(stores, st.session_state.last_event)

        if st.session_state.last_run:
            res = st.session_state.last_run
            left, right = st.columns([1.2, 1])
            with left:
                st.write("### Step-by-step trace")
                for item in res["ui_trace"]:
                    with st.expander(item["step"], expanded=False):
                        st.json(item["output"])
            with right:
                st.write("### Final Maintenance Recommendation Packet")
                st.json(res["final_packet"])

with tab3:
    st.subheader("Approval Inbox (Policy Gate simulation)")
    pending = [a for a in stores["approvals"].values() if a["status"] == "PENDING"]
    approved = [a for a in stores["approvals"].values() if a["status"] == "APPROVED"]
    st.write(f"Pending: {len(pending)} | Approved: {len(approved)}")

    if pending:
        for appr in pending:
            with st.expander(f"{appr['token']} — {appr['asset_id']} — {appr['recommended_action']}", expanded=False):
                st.json(appr)
                if st.button(f"Approve {appr['token']}", key=f"approve-{appr['token']}"):
                    stores["approvals"][appr["token"]]["status"] = "APPROVED"
                    st.success("Approved.")
    else:
        st.info("No pending approvals right now.")

with tab4:
    st.subheader("Audit Log + Work Orders")
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Work Orders (CMMS)")
        st.json(stores["work_orders"] if stores["work_orders"] else {"note": "none yet"})
    with col2:
        st.write("### Audit Entries (last 5)")
        st.json(stores["audit"][-5:] if stores["audit"] else {"note": "none yet"})
    st.write("### Notifications (last 10)")
    st.json(stores["notifications"][-10:] if stores["notifications"] else {"note": "none yet"})