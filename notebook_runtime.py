"""Explicit paths and authentication shared by desktop/Colab notebooks."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def data_dir():
    """Set RMT_DATA_DIR to the existing research data folder if required."""
    path = Path(os.environ.get("RMT_DATA_DIR", ROOT / "outputs" / "data")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_dir():
    path = Path(os.environ.get("RMT_OUTPUT_DIR", ROOT / "outputs")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_auth():
    """Prefer HF_TOKEN or the Hugging Face login cache; Colab secrets are optional.

    VS Code's remote Colab kernel may not expose the browser secrets API.
    In that case run huggingface_hub.login() interactively in that kernel.
    """
    from huggingface_hub import get_token
    if get_token():
        return
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
    except Exception:
        return
    if token:
        os.environ["HF_TOKEN"] = token
