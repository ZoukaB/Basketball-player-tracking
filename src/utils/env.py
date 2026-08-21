"""Load Roboflow / HuggingFace keys from a local ``.env`` file."""

from __future__ import annotations

import os
from pathlib import Path


def load_api_keys(env_path: str | Path | None = None) -> None:
    """Read ``.env`` and copy ``ROBOFLOW_API_KEY`` to ``API_KEY``.

    Roboflow ``inference.get_model`` looks up ``API_KEY``. The notebook stored
    the secret as ``ROBOFLOW_API_KEY``, so we set both.
    """
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise ImportError(
            "python-dotenv is required to load .env. "
            "pip install python-dotenv"
        ) from exc

    if env_path is None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(env_path, override=False)

    roboflow_key = os.environ.get("ROBOFLOW_API_KEY")
    if roboflow_key:
        os.environ.setdefault("API_KEY", roboflow_key)

    if not os.environ.get("API_KEY"):
        raise RuntimeError(
            "Missing ROBOFLOW_API_KEY. Create a .env at the repo root with "
            "ROBOFLOW_API_KEY=... and HF_TOKEN=..."
        )
