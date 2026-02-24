from __future__ import annotations

import os
from dotenv import load_dotenv
from google import genai

# load .env from project root automatically
load_dotenv()

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


class GeminiClient:
    def __init__(self, model: str | None = None):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

        self.model = model or DEFAULT_MODEL
        self.client = genai.Client(api_key=api_key)

    def generate_text(self, prompt: str, temperature: float = 0.2) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"temperature": temperature},
        )
        return (resp.text or "").strip()

    def generate_json(self, system_prompt: str, payload: dict, temperature: float = 0.2) -> dict:
        """
        If your earlier version already had JSON parsing, keep it.
        Minimal safe approach: ask for JSON and try to parse.
        """
        import json

        full_prompt = (
            system_prompt
            + "\n\nReturn ONLY valid RFC8259 JSON.\n\nINPUT:\n"
            + json.dumps(payload, ensure_ascii=False)
        )

        text = self.generate_text(full_prompt, temperature=temperature)
        try:
            return json.loads(text)
        except Exception:
            return {"error": "INVALID_JSON", "raw": text}