from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


def get_client():
    """Load OpenAI credentials and return (client, assistant_id)."""
    # Load .env from current or parent directories and user's Desktop
    env_path = find_dotenv(usecwd=True)
    if env_path:
        load_dotenv(env_path, override=True)
    desktop_env = Path.home() / "Desktop" / ".env"
    if desktop_env.exists():
        load_dotenv(desktop_env, override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    assistant_id = os.getenv("ASSISTANT_ID") or os.getenv("OPENAI_ASSISTANT_ID")
    if not api_key:
        raise ValueError("🚨 Missing OPENAI_API_KEY in environment.")
    if not assistant_id:
        raise ValueError("🚨 Missing ASSISTANT_ID in environment.")

    client = OpenAI(api_key=api_key)
    return client, assistant_id
