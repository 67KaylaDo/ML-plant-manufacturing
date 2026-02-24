from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

@dataclass(frozen=True)
class Settings:
    app_name: str = "ML-plant-manufacturing"
    data_dir: str = "data"
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

settings = Settings()