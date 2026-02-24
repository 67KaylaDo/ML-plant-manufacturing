from __future__ import annotations

from typing import Any
from ml_plant.llm_gemini import GeminiClient

COPILOT_PROMPT = """You are Titan's Agentic AI Copilot for the Predictive Maintenance MVP demo.

Rules:
- You must ground all answers ONLY in the provided context payload (final_packet, ui_trace, approvals, work_orders, notifications).
- If the user asks for something not present in context, say what is missing and suggest running the workflow.
- Be concise, structured, and practical.
- Do NOT invent telemetry, tickets, model outputs, or events.

Return plain text (not JSON).
"""

def answer_question(question: str, context: dict[str, Any]) -> str:
    llm = GeminiClient()

    # We reuse the JSON generator but ask for plain text. Easiest: call generate_json is too strict.
    # So we call GeminiClient client directly via a tiny hack: use generate_json but wrap output.
    # Instead, simplest: create a lightweight text call using the underlying client.
    # We'll do that here without changing llm_gemini.py too much.

    user_payload = {
        "question": question,
        "context": context,
    }

    # Use the GeminiClient internals safely:
    resp = llm.client.models.generate_content(
        model=llm.model,
        contents=[{"role": "user", "parts": [{"text": COPILOT_PROMPT + "\n\n" + str(user_payload)}]}],
        config={"temperature": 0.2},
    )

    text = (resp.text or "").strip()
    if not text:
        return "I couldn't generate a response. Try again, or re-run the workflow."
    return text