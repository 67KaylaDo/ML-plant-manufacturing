from __future__ import annotations

import json
from typing import Any
from google import genai
from ml_plant.config import settings

class GeminiClient:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    def generate_json(self, prompt: str, payload: dict[str, Any], temperature: float = 0.2) -> dict[str, Any]:
        user_text = "Return ONLY RFC8259 JSON.\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)

        resp = self.client.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": [{"text": prompt + "\n\n" + user_text}]}],
            config={"temperature": temperature},
        )

        text = (resp.text or "").strip()

        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
            raise ValueError(f"Model did not return valid JSON:\n{text}")