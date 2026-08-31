"""Load Roboflow / HuggingFace keys from a local ``.env`` file."""

from __future__ import annotations

import os
from pathlib import Path


def configure_ssl() -> None:
    """Trust OS certificates so Avast/corporate HTTPS inspection works.

    Conda often sets ``SSL_CERT_FILE`` to its own ``cacert.pem``. Python
    ``requests`` then ignores the Windows store, where Avast's intercept
    root lives, and Roboflow calls fail with CERTIFICATE_VERIFY_FAILED.
    Call this before importing ``inference``.
    """
    def _is_conda_bundle(path: str) -> bool:
        lowered = path.replace("\\", "/").lower()
        return any(marker in lowered for marker in ("anaconda", "miniconda", "/envs/", "\\envs\\"))

    ssl_cert_file = os.environ.get("SSL_CERT_FILE", "")
    requests_ca = os.environ.get("REQUESTS_CA_BUNDLE", "")
    if _is_conda_bundle(ssl_cert_file):
        os.environ.pop("SSL_CERT_FILE", None)
    if _is_conda_bundle(requests_ca):
        os.environ.pop("REQUESTS_CA_BUNDLE", None)

    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        pass


def load_api_keys(env_path: str | Path | None = None) -> None:
    """Read ``.env`` and copy ``ROBOFLOW_API_KEY`` to ``API_KEY``.

    Roboflow ``inference.get_model`` looks up ``API_KEY``. The notebook stored
    the secret as ``ROBOFLOW_API_KEY``, so we set both.
    """
    configure_ssl()
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

    roboflow_key = _clean_secret(os.environ.get("ROBOFLOW_API_KEY"))
    if roboflow_key:
        os.environ["ROBOFLOW_API_KEY"] = roboflow_key
        # inference.get_model reads API_KEY at import time; always overwrite
        # a stale/empty value so later calls can pass the live key.
        os.environ["API_KEY"] = roboflow_key

    hf_token = _clean_secret(os.environ.get("HF_TOKEN"))
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    if not os.environ.get("API_KEY"):
        raise RuntimeError(
            "Missing ROBOFLOW_API_KEY. Create a .env at the repo root with "
            "ROBOFLOW_API_KEY=... and HF_TOKEN=..."
        )


def _clean_secret(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().strip('"').strip("'")


def roboflow_api_key() -> str:
    """Return the Roboflow key, loading ``.env`` if needed."""
    load_api_keys()
    return os.environ["API_KEY"]
