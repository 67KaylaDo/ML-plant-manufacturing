from __future__ import annotations

from typing import Any, Dict, List


COPILOT_SYSTEM_PROMPT = """You are Titan's Agentic AI Copilot for the Predictive Maintenance MVP demo.

Rules:
- You MUST ground all answers ONLY in the provided context payload:
  (final_packet, ui_trace, approvals, work_orders, notifications, audit_log)
- If the user asks for something not present in context, say what is missing and suggest running the workflow.
- Be concise, structured, and practical.
- Do NOT invent telemetry, tickets, model outputs, or events.
- Return plain text (not JSON). No markdown.
"""


def _safe_get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def _summarize_without_llm(context: Dict[str, Any], question: str) -> str:
    """
    Deterministic fallback if Gemini is unavailable.
    """
    final_packet = _safe_get(context, "final_packet", {})
    approvals = _safe_get(context, "approvals", [])
    work_orders = _safe_get(context, "work_orders", {})
    notifications = _safe_get(context, "notifications", [])

    # Extract assets from final packet
    assets = final_packet.get("assets") or []
    if isinstance(assets, dict):
        assets = list(assets.values())

    # Pending approvals
    pending = []
    for a in approvals if isinstance(approvals, list) else []:
        if (a.get("status") or "").upper() == "PENDING":
            pending.append(a)

    # Basic answers based on question intent
    q = (question or "").lower()

    if "approval" in q:
        if not pending:
            return "No pending approvals found in context. Run the workflow with at least 2 assets selected to force an approval for the 2nd asset."
        lines = ["Pending approvals:"]
        for item in pending[:10]:
            lines.append(
                f"- asset_id={item.get('asset_id')} action={item.get('recommended_action')} token={item.get('token')} status={item.get('status')}"
            )
        return "\n".join(lines)

    if "evidence" in q or "strongest" in q or "highest risk" in q:
        if not assets:
            return "No final recommendation packet found in context. Please run the workflow first."
        # pick asset with lowest confidence or any heuristic; we prefer ones requiring approval
        risky = None
        for a in assets:
            if a.get("requires_human_approval"):
                risky = a
                break
        if risky is None:
            risky = assets[0]

        ev = risky.get("evidence", {})
        return (
            "Strongest available evidence (from context):\n"
            f"- asset_id: {risky.get('asset_id')}\n"
            f"- recommended_action: {risky.get('recommended_action')}\n"
            f"- confidence_score: {risky.get('confidence_score')}\n"
            f"- evidence pointers: {ev}"
        )

    # default: explain recommendation packet
    if not final_packet:
        return "No final recommendation packet found in context. Please run the workflow first."

    lines = []
    lines.append(f"Event: {final_packet.get('event_id')} | Plant: {final_packet.get('plant_id')}")
    lines.append(f"Overall status: {final_packet.get('overall_status')} | Audit trace: {final_packet.get('audit_trace_id')}")
    lines.append("Recommendations:")
    if not assets:
        lines.append("- (no assets in packet)")
    else:
        for a in assets[:10]:
            lines.append(
                f"- {a.get('asset_id')}: action={a.get('recommended_action')}, "
                f"approval={a.get('requires_human_approval')}, conf={a.get('confidence_score')}"
            )

    # work orders summary
    if isinstance(work_orders, dict) and work_orders:
        lines.append(f"Work orders created: {len(work_orders)}")
    else:
        lines.append("Work orders created: 0 (or pending approvals)")

    # notifications summary
    if isinstance(notifications, list) and notifications:
        lines.append(f"Notifications sent: {len(notifications)}")
    else:
        lines.append("Notifications sent: 0")

    # approvals summary
    if pending:
        lines.append(f"Pending approvals: {len(pending)}")
    else:
        lines.append("Pending approvals: 0")

    return "\n".join(lines)


def answer_question(question: str, context: Dict[str, Any]) -> str:
    """
    Grounded chat for the Streamlit Copilot tab.

    Tries Gemini first. If missing key / Gemini errors, returns deterministic fallback answer.
    """
    # Build a compact payload (avoid huge ui_trace)
    final_packet = _safe_get(context, "final_packet", {})
    approvals = _safe_get(context, "approvals", [])
    work_orders = _safe_get(context, "work_orders", {})
    notifications = _safe_get(context, "notifications", [])
    audit_log = _safe_get(context, "audit_log", [])

    # Keep last N trace steps if present
    ui_trace = _safe_get(context, "ui_trace", [])
    if isinstance(ui_trace, list) and len(ui_trace) > 12:
        ui_trace = ui_trace[-12:]

    payload = {
        "question": question,
        "final_packet": final_packet,
        "approvals": approvals,
        "work_orders": work_orders,
        "notifications": notifications,
        "audit_log": audit_log[-5:] if isinstance(audit_log, list) else audit_log,
        "ui_trace": ui_trace,
    }

    # Try Gemini
    try:
        from ml_plant.llm_gemini import GeminiClient

        llm = GeminiClient()
        prompt = (
            COPILOT_SYSTEM_PROMPT
            + "\n\nQUESTION:\n"
            + (question or "")
            + "\n\nCONTEXT (JSON-like):\n"
            + str(payload)
        )
        text = llm.generate_text(prompt, temperature=0.2)
        text = (text or "").strip()
        if text:
            return text
        return "I couldn't generate a response. Try again, or re-run the workflow."
    except Exception:
        # Deterministic fallback
        return _summarize_without_llm(payload, question)