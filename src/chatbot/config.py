"""Settings that come from the environment (or `.env`)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "chatbot.db"
DEFAULT_MODEL = "deepseek-flash"

SYSTEM_PROMPT = (
    "You are a friendly, concise assistant. "
    "Answer in plain language and keep replies short unless the user asks for details."
)


def db_path() -> Path:
    """Absolute path of the SQLite file, created on demand."""
    raw = os.getenv("CHATBOT_DB_PATH")
    path = Path(raw) if raw else DEFAULT_DB_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def model_name() -> str:
    return os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)


def missing_api_key() -> bool:
    return not os.getenv("DEEPSEEK_API_KEY")
