from __future__ import annotations

import os
import json
from dotenv import load_dotenv
from google import genai

# --------------------------------------------------
# Load .env automatically
# --------------------------------------------------
load_dotenv()

# Allow override via .env
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Optional fallback list (in case Google renames models again)
FALLBACK_MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
]


class GeminiClient:
    def __init__(self, model: str | None = None):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        self.model = model or DEFAULT_MODEL
        self.client = genai.Client(api_key=api_key)

    # --------------------------------------------------
    # Safe model call with automatic fallback
    # --------------------------------------------------
    def _safe_generate(self, model: str, prompt: str, temperature: float):
        return self.client.models.generate_content(
            model=model,
            contents=prompt,
            config={"temperature": temperature},
        )

    def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        """
        Safe text generation with fallback models.
        Prevents 404 model crashes.
        """
        models_to_try = [self.model] + [
            m for m in FALLBACK_MODELS if m != self.model
        ]

        last_error = None

        for m in models_to_try:
            try:
                resp = self._safe_generate(m, prompt, temperature)
                return (resp.text or "").strip()
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

    # --------------------------------------------------
    # Robust JSON generation
    # --------------------------------------------------
    def generate_json(
        self,
        system_prompt: str,
        payload: dict,
        temperature: float = 0.2,
    ) -> dict:
        """
        Forces Gemini to return JSON.
        Cleans markdown blocks if present.
        """

        full_prompt = (
            system_prompt
            + "\n\nReturn ONLY valid RFC8259 JSON.\n\nINPUT:\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        text = self.generate_text(full_prompt, temperature=temperature)

        if not text:
            return {"error": "EMPTY_RESPONSE"}

        # Strip markdown ```json blocks if Gemini adds them
        text = text.strip()
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        try:
            return json.loads(text)
        except Exception:
            return {
                "error": "INVALID_JSON",
                "raw": text,
            }