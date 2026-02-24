from __future__ import annotations
from typing import Any
import re

try:
    from ml_plant.llm_gemini import GeminiClient
    HAS_GEMINI = True
except Exception:
    GeminiClient = None
    HAS_GEMINI = False


def _is_greeting(q: str) -> bool:
    q = q.strip().lower()
    return q in {"hi", "hello", "hey", "yo"} or q.startswith(("hi ", "hello ", "hey "))


def _extract_id(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1) if m else None


def _format_pending_approvals(approvals: list[dict]) -> str:
    pending = [a for a in approvals if a.get("status") == "PENDING"]
    if not pending:
        return "No pending approvals right now."

    lines = ["Pending approvals:"]
    for a in pending:
        lines.append(
            f"- {a.get('token')} | asset={a.get('asset_id')} | action={a.get('recommended_action')} "
            f"| reason={a.get('expected_impact', {}).get('risk_reduction_summary','(no reason provided)')}"
        )
    return "\n".join(lines)


def _find_alert(alert_id: str, context: dict[str, Any]) -> dict | None:
    alerts = context.get("alerts_preview") or []
    for a in alerts:
        if a.get("alert_id") == alert_id:
            return a
    return None


def _find_event(event_id: str, context: dict[str, Any]) -> dict | None:
    ev = context.get("last_event")
    if isinstance(ev, dict) and ev.get("event_id") == event_id:
        return ev
    # also try in ui_trace if present
    ui_trace = context.get("ui_trace") or []
    for step in ui_trace:
        out = step.get("output") if isinstance(step, dict) else None
        if isinstance(out, dict):
            maybe_event = out.get("event") or out.get("payload", {}).get("event")
            if isinstance(maybe_event, dict) and maybe_event.get("event_id") == event_id:
                return maybe_event
    return None


def answer_question(question: str, context: dict[str, Any]) -> str:
    q = (question or "").strip()
    if not q:
        return "Ask me about the workflow results, approvals, alerts, or recommendations."

    # 1) greetings
    if _is_greeting(q):
        return (
            "Hi! I’m the grounded Copilot for this MVP.\n"
            "Try:\n"
            "- “Explain latest recommendation”\n"
            "- “What needs approval?”\n"
            "- “Explain event evt-xxxx”\n"
            "- “Explain alert cdd8ea85”"
        )

    approvals = context.get("approvals") or []
    final_packet = context.get("final_packet")
    last_event = context.get("last_event")

    # 2) explicit approval questions
    if "approval" in q.lower() or "approve" in q.lower() or "pending" in q.lower():
        return _format_pending_approvals(approvals)

    # 3) explain specific event
    event_id = _extract_id(r"(evt-[a-z0-9]+)", q)
    if event_id:
        ev = _find_event(event_id, context)
        if not ev:
            return f"I can’t find event **{event_id}** in context. Run Tab 1 → Create SensorAlertEvent first."
        return (
            f"Event {ev.get('event_id')} ({ev.get('event_type')}):\n"
            f"- plant_id: {ev.get('plant_id')}\n"
            f"- start_time: {ev.get('start_time')}\n"
            f"- end_time: {ev.get('end_time')}\n"
            f"- next step: run Tab 2 workflow to generate recommendations."
        )

    # 4) explain specific alert_id
    alert_id = _extract_id(r"(?:alert[_\s]?id[:=\s\"']+)?([a-z0-9]{6,})", q)
    if "alert" in q.lower() and alert_id:
        a = _find_alert(alert_id, context)
        if not a:
            return (
                f"I can’t find alert_id **{alert_id}** in the current preview.\n"
                f"Tip: Tab 1 shows first 10 alerts; increase preview or rerun event."
            )
        return (
            f"Alert {a.get('alert_id')}:\n"
            f"- asset_id: {a.get('asset_id')}\n"
            f"- sensor: {a.get('sensor')}\n"
            f"- code: {a.get('code')}\n"
            f"- severity: {a.get('severity')}\n"
            f"- value: {a.get('value')}\n"
            f"- ts: {a.get('ts')}"
        )

    # 5) explain latest recommendation packet (grounded)
    if "recommendation" in q.lower() or "latest" in q.lower() or "packet" in q.lower():
        if not final_packet or isinstance(final_packet, dict) and final_packet.get("error"):
            return "No final recommendation packet found. Run Tab 2 workflow first."
        assets = final_packet.get("assets", [])
        lines = [
            f"Latest packet:",
            f"- event_id: {final_packet.get('event_id')}",
            f"- plant_id: {final_packet.get('plant_id')}",
            f"- overall_status: {final_packet.get('overall_status')}",
            f"- audit_trace_id: {final_packet.get('audit_trace_id')}",
            "",
            "Recommendations:"
        ]
        for a in assets:
            lines.append(
                f"- {a.get('asset_id')}: action={a.get('recommended_action')}, "
                f"approval={a.get('requires_human_approval')}, conf={a.get('confidence_score')}"
            )
        return "\n".join(lines)

    # 6) if Gemini available, use it as fallback; otherwise return safe help
    if HAS_GEMINI and GeminiClient is not None:
        try:
            llm = GeminiClient()
            prompt = (
                "You are a grounded assistant. Answer ONLY using this context.\n"
                "If missing, say what is missing.\n\n"
                f"QUESTION:\n{q}\n\nCONTEXT:\n{context}"
            )
            # using llm.generate if your client provides it
            return llm.generate(prompt)
        except Exception:
            pass

    # final fallback
    return (
        "I can answer questions about:\n"
        "- approvals (pending approvals)\n"
        "- latest recommendation packet\n"
        "- specific event_id (evt-xxxx)\n"
        "- alert_id (from preview)\n\n"
        "Try: “What needs approval?” or run Tab 2 first."
    )