from __future__ import annotations

from typing import Any, Dict, List, Optional
import re

try:
    from ml_plant.llm_gemini import GeminiClient
    HAS_GEMINI = True
except Exception:
    GeminiClient = None
    HAS_GEMINI = False


# ============================================================
# Grounded Copilot System Prompt (VISIBLE IN CODE)
# ============================================================
COPILOT_PROMPT = """You are the grounded Copilot for ML-plant-manufacturing MVP.

Rules:
- Use ONLY the provided context.
- If missing data, say what is missing and suggest which tab/workflow step to run.
- Never invent alerts, events, approvals, telemetry, or model results.
- Be concise and structured.
- Return plain text (not JSON).

If user greets, explain how to use the system.
"""


# ============================================================
# Intent / parsing helpers
# ============================================================
_GREET_RE = re.compile(r"^\s*(hi|hello|hey|yo|sup|good\s*(morning|afternoon|evening))\s*[!.]*\s*$", re.I)
_EVT_RE = re.compile(r"\b(evt-[a-z0-9]+)\b", re.I)

# alert_id in your synthetic data is often hex-like (6–12), but allow longer too
# also supports patterns like: alert_id:"abcd1234"
_ALERT_RE = re.compile(r"(?:alert[_\s-]*id\s*[:=]\s*)?\"?([a-z0-9\-]{6,32})\"?", re.I)


def _norm(s: str) -> str:
    return (s or "").strip()


def _is_greeting(q: str) -> bool:
    return bool(_GREET_RE.match(q))


def _extract_event_id(text: str) -> Optional[str]:
    m = _EVT_RE.search(text or "")
    return m.group(1) if m else None


def _extract_alert_id(text: str) -> Optional[str]:
    """
    Extract alert_id if user types:
    - explain alert_id: abcd1234
    - explain alert abcd1234
    - alert_id="abcd1234"
    If user just types a random word, we don't want to treat it as alert_id.
    So we only use this if the question contains the word 'alert'.
    """
    if "alert" not in (text or "").lower():
        return None
    m = _ALERT_RE.search(text or "")
    return m.group(1) if m else None


# ============================================================
# Context extraction (robust to list/dict)
# ============================================================
def _get_approvals(context: Dict[str, Any]) -> List[dict]:
    approvals = context.get("approvals") or []
    if isinstance(approvals, dict):
        approvals = list(approvals.values())
    if not isinstance(approvals, list):
        return []
    return [a for a in approvals if isinstance(a, dict)]


def _pending_approvals(context: Dict[str, Any]) -> List[dict]:
    approvals = _get_approvals(context)
    pending = []
    for a in approvals:
        status = str(a.get("status", "PENDING")).upper()
        if status == "PENDING":
            pending.append(a)
    return pending


def _get_final_packet(context: Dict[str, Any]) -> dict:
    pkt = context.get("final_packet") or {}
    return pkt if isinstance(pkt, dict) else {}


def _get_last_event(context: Dict[str, Any]) -> dict:
    ev = context.get("last_event") or {}
    return ev if isinstance(ev, dict) else {}


def _get_ui_trace(context: Dict[str, Any]) -> List[dict]:
    tr = context.get("ui_trace") or []
    return tr if isinstance(tr, list) else []


def _get_alerts_preview(context: Dict[str, Any]) -> List[dict]:
    alerts = context.get("alerts_preview") or []
    if isinstance(alerts, list):
        return [a for a in alerts if isinstance(a, dict)]
    return []


# ============================================================
# Grounded response builders
# ============================================================
def _help_text() -> str:
    return (
        "I’m the grounded Copilot for this MVP.\n\n"
        "Try:\n"
        "- Explain latest recommendation\n"
        "- What needs approval?\n"
        "- Explain event evt-xxxx\n"
        "- Explain alert_id <id>\n\n"
        "If you haven’t run the workflow: Tab 1 → Create event, Tab 2 → Run workflow."
    )


def _format_pending_approvals(pending: List[dict]) -> str:
    if not pending:
        return (
            "No pending approvals found in the current Copilot context.\n"
            "If you expected approvals: run Tab 2 with ≥2 assets selected (demo rule forces approval on 2nd asset), "
            "then check Tab 3."
        )

    lines = ["Pending approvals (from context):"]
    for a in pending:
        token = a.get("token") or a.get("approval_token") or "(no token)"
        asset_id = a.get("asset_id", "(unknown asset)")
        action = a.get("recommended_action", "(unknown action)")
        reason = ""
        impact = a.get("expected_impact") or {}
        if isinstance(impact, dict):
            reason = impact.get("risk_reduction_summary") or ""
        if reason:
            lines.append(f"- {token} | asset={asset_id} | action={action} | reason={reason}")
        else:
            lines.append(f"- {token} | asset={asset_id} | action={action}")
    lines.append("")
    lines.append("To proceed: approve in Tab 3 (Approval Inbox).")
    return "\n".join(lines)


def _format_latest_packet(context: Dict[str, Any]) -> str:
    pkt = _get_final_packet(context)
    if not pkt:
        return "No final recommendation packet found. Run Tab 2 workflow first."

    if pkt.get("error"):
        return f"Workflow did not produce a final packet. Error: {pkt.get('error')}"

    assets = pkt.get("assets", [])
    if not isinstance(assets, list):
        assets = []

    lines = [
        "Latest recommendation packet:",
        f"- event_id: {pkt.get('event_id')}",
        f"- plant_id: {pkt.get('plant_id')}",
        f"- overall_status: {pkt.get('overall_status')}",
        f"- audit_trace_id: {pkt.get('audit_trace_id')}",
        "",
        "Recommendations:",
    ]
    if not assets:
        lines.append("- (no assets in packet)")
        return "\n".join(lines)

    for a in assets:
        if not isinstance(a, dict):
            continue
        lines.append(
            f"- {a.get('asset_id')}: action={a.get('recommended_action')}, "
            f"approval={a.get('requires_human_approval')}, conf={a.get('confidence_score')}"
        )
    return "\n".join(lines)


def _format_event(event_id: str, context: Dict[str, Any]) -> str:
    ev = _get_last_event(context)
    if not ev:
        return "No event found in context. Create an event in Tab 1 first."

    if ev.get("event_id") != event_id:
        return (
            f"I only have context for the latest event: {ev.get('event_id')}.\n"
            f"You asked about: {event_id}.\n\n"
            "To explain that event, create/run it again so it becomes the active event (Tab 1 → Tab 2)."
        )

    return (
        f"Event {ev.get('event_id')} ({ev.get('event_type')}):\n"
        f"- plant_id: {ev.get('plant_id')}\n"
        f"- start_time: {ev.get('start_time')}\n"
        f"- end_time: {ev.get('end_time')}\n\n"
        "This event is the trigger that starts the multi-agent workflow (T1 → A2 → A3 → A4 → Policy → CMMS/Audit)."
    )


def _find_alert_in_ui_trace(alert_id: str, ui_trace: List[dict]) -> Optional[dict]:
    """
    Search for alert inside the T1 QueryAlertStore output, if present.
    """
    for step in ui_trace:
        if not isinstance(step, dict):
            continue
        out = step.get("output")
        if not isinstance(out, dict):
            continue
        alerts = out.get("alerts")
        if not isinstance(alerts, list):
            continue
        for a in alerts:
            if isinstance(a, dict) and str(a.get("alert_id")) == str(alert_id):
                return a
    return None


def _format_alert(alert_id: str, context: Dict[str, Any]) -> str:
    # search in preview first
    for a in _get_alerts_preview(context):
        if str(a.get("alert_id")) == str(alert_id):
            return (
                f"Alert {a.get('alert_id')}:\n"
                f"- asset_id: {a.get('asset_id')}\n"
                f"- sensor: {a.get('sensor')}\n"
                f"- code: {a.get('code')}\n"
                f"- severity: {a.get('severity')}\n"
                f"- value: {a.get('value')}\n"
                f"- ts: {a.get('ts')}\n\n"
                "Note: Raw alerts are clustered by the triage agent (A2) to reduce noise."
            )

    # search in ui_trace (T1 output)
    ui_trace = _get_ui_trace(context)
    found = _find_alert_in_ui_trace(alert_id, ui_trace)
    if found:
        return (
            f"Alert {found.get('alert_id')} (from T1 QueryAlertStore):\n"
            f"- asset_id: {found.get('asset_id')}\n"
            f"- sensor: {found.get('sensor')}\n"
            f"- code: {found.get('code')}\n"
            f"- severity: {found.get('severity')}\n"
            f"- value: {found.get('value')}\n"
            f"- ts: {found.get('ts')}\n\n"
            "Next: Ask “Explain latest recommendation” to see the final decision this alert contributed to."
        )

    return (
        f"I can’t find alert_id {alert_id} in the current context.\n"
        "Tip: Copy an alert_id from Tab 1 preview (first 10 alerts), then ask again."
    )


# ============================================================
# Main entry
# ============================================================
def answer_question(question: str, context: Dict[str, Any]) -> str:
    q_raw = _norm(question)
    if not q_raw:
        return _help_text()

    q_low = q_raw.lower()

    # 1) Greetings
    if _is_greeting(q_raw):
        return _help_text()

    # 2) Approvals
    if any(k in q_low for k in ["approval", "approve", "pending"]):
        return _format_pending_approvals(_pending_approvals(context))

    # 3) Explain latest packet
    if any(k in q_low for k in ["latest recommendation", "latest packet", "recommendation", "packet", "summary"]):
        return _format_latest_packet(context)

    # 4) Explain event evt-xxxx
    evt_id = _extract_event_id(q_raw)
    if evt_id and ("event" in q_low or "explain" in q_low):
        return _format_event(evt_id, context)

    # 5) Explain alert alert_id
    alert_id = _extract_alert_id(q_raw)
    if alert_id:
        return _format_alert(alert_id, context)

    # 6) Gemini fallback (ONLY when needed)
    # Still grounded by COPILOT_PROMPT.
    if HAS_GEMINI and GeminiClient is not None:
        try:
            llm = GeminiClient()
            prompt = (
                COPILOT_PROMPT
                + "\n\nQUESTION:\n"
                + q_raw
                + "\n\nCONTEXT:\n"
                + str(context)
            )
            return llm.generate(prompt)
        except Exception:
            pass

    # 7) Safe fallback
    return (
        "I can help with:\n"
        "- approvals (pending approvals)\n"
        "- latest recommendation packet\n"
        "- explain event evt-xxxx\n"
        "- explain alert_id <id>\n\n"
        "Tip: run Tab 1 → create event, Tab 2 → run workflow."
    )