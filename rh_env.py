"""Read `.env` so a local run and a compose run see the same configuration.

compose reads `.env` on its own and turns it into container environment, so a
setting put there reached the deployment but not a local `uvicorn rh_server:app`
- the same file, silently in effect on one side only. Loading it here removes the
gap: whatever configures docker configures a dev run too.

Real environment always wins, so `RH_X=1 uvicorn ...` and compose's `environment:`
block still override the file.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List
DEFAULT_ENV_FILE = Path(__file__).parent / ".env"
def parse(text: str) -> Dict[str, str]:
    """Parse the subset of dotenv syntax compose itself accepts.

    Deliberately no interpolation and no `export` prefix: compose does not expand
    `${VAR}` inside `.env` either, so supporting it here would make the two sides
    disagree again in exactly the way this module exists to prevent.
    """
    out: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip one layer of matching quotes; a quoted value keeps inner whitespace.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        out[key] = value
    return out
def load(path: Path = DEFAULT_ENV_FILE) -> List[str]:
    """Apply `.env` to os.environ without overwriting what is already set."""
    path = Path(path)
    if not path.exists():
        return []
    try:
        pairs = parse(path.read_text(encoding="utf-8"))
    except Exception:
        # A malformed .env must not stop the server from starting; the settings it
        # carries all have defaults.
        return []
    applied: List[str] = []
    for key, value in pairs.items():
        if key in os.environ:
            continue
        os.environ[key] = value
        applied.append(key)
    return applied
