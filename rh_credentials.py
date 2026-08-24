from __future__ import annotations
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0"
DEFAULT_OUTPUT = Path(__file__).parent / "credentials.json"
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
@dataclass
class Credential:
    id: str
    cookie_header: str = ""
    user_agent: str = DEFAULT_USER_AGENT
    created_at: str = field(default_factory=_utc_now)
    source: str = "anonymous"
    def is_valid(self) -> bool:
        return True
    def cookie_dict(self) -> dict:
        out: dict = {}
        for part in self.cookie_header.split(";"):
            if "=" in part:
                k, v = part.strip().split("=", 1)
                out[k] = v
        return out
class CredentialFile:
    def __init__(self, path: Path = DEFAULT_OUTPUT) -> None:
        self.path = Path(path)
    def load(self) -> List[Credential]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        items = raw.get("credentials", []) if isinstance(raw, dict) else raw
        creds: List[Credential] = []
        for item in items:
            creds.append(Credential(
                id=item.get("id") or str(uuid.uuid4()),
                cookie_header=item.get("cookie_header", ""),
                user_agent=item.get("user_agent", DEFAULT_USER_AGENT),
                created_at=item.get("created_at", _utc_now()),
                source=item.get("source", "anonymous"),
            ))
        return creds
    def save(self, credentials: List[Credential]) -> None:
        payload = {
            "version": 1,
            "updated_at": _utc_now(),
            "count": len(credentials),
            "credentials": [asdict(c) for c in credentials],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
def ensure_fresh(credential: Credential) -> Credential:
    return credential