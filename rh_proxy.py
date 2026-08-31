"""Egress proxy selection with sticky sessions, addressed by platform + account.

Speaks the forward-proxy convention used by Resin: the proxy username carries the
identity as ``Platform.Account`` and the password is the shared proxy token. The
platform picks the node pool, the account is what the upstream pins a stable
egress IP to, so the same account keeps the same IP across requests.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote, urlsplit
DEFAULT_CONFIG = Path(__file__).parent / "proxy.json"
DEFAULT_PLATFORM = "Default"
STICKY_MODES = ("credential", "client", "fixed", "none")
ALLOWED_SCHEMES = ("http", "https", "socks5", "socks5h", "socks4", "socks4a")
CHECK_URL = "https://api.ipify.org?format=json"
CHECK_TIMEOUT = (10, 20)
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
class ProxyUnavailableError(RuntimeError):
    """The egress proxy itself failed. Not the fault of the upstream credential."""
def _tag(value: Any, fallback: str = "") -> str:
    """Reduce to characters safe both in URL userinfo and in the Platform.Account form."""
    out = _UNSAFE.sub("_", str(value or "")).strip("_")
    return out or fallback
def _env_str(name: str) -> Optional[str]:
    """Empty counts as unset, so compose passing ${VAR:-} cannot wipe the file config.
    An intentionally blank token belongs in proxy.json as "token": ""."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip() or None
def _env_bool(name: str) -> Optional[bool]:
    raw = _env_str(name)
    if not raw:
        return None
    low = raw.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    return None
def caller_tag(auth_header: Optional[str], peer: Optional[str]) -> str:
    """Stable per-caller tag: prefer the downstream API key, fall back to the peer address."""
    basis = (auth_header or "").strip() or (peer or "").strip()
    if not basis:
        return ""
    return "c" + hashlib.sha256(basis.encode("utf-8", "replace")).hexdigest()[:12]
def _normalize_url(raw: str) -> str:
    url = (raw or "").strip()
    if not url:
        return ""
    if "://" not in url:
        url = "http://" + url
    parts = urlsplit(url)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"unsupported proxy scheme {parts.scheme!r}; use one of {', '.join(ALLOWED_SCHEMES)}")
    if not parts.hostname:
        raise ValueError(f"proxy url {raw!r} has no host")
    if parts.scheme.startswith("socks"):
        try:
            import socks  # noqa: F401  (PySocks, pulled in by requests[socks])
        except ImportError as exc:
            raise ValueError(f"{parts.scheme} proxies need PySocks: pip install 'requests[socks]'") from exc
    port = f":{parts.port}" if parts.port else ""
    # Drop any userinfo baked into the url; identity is attached per request instead.
    return f"{parts.scheme}://{parts.hostname}{port}"
@dataclass
class ProxyConfig:
    enabled: bool = False
    url: str = ""
    token: str = ""
    platform: str = DEFAULT_PLATFORM
    account: str = ""
    sticky: str = "credential"
    trust_headers: bool = True
    verify_tls: bool = True
    bindings: Dict[str, Dict[str, str]] = field(default_factory=dict)
    @property
    def active(self) -> bool:
        return bool(self.enabled and self.url)
def _parse_binding(value: Any) -> Dict[str, str]:
    """Accept either {"platform": .., "account": ..} or the shorthand "Platform.Account"."""
    if isinstance(value, dict):
        return {"platform": _tag(value.get("platform")), "account": _tag(value.get("account"))}
    text = str(value or "").strip()
    if "." in text:
        head, _, tail = text.partition(".")
        return {"platform": _tag(head), "account": _tag(tail)}
    return {"platform": "", "account": _tag(text)}
def load_config(path: Path = DEFAULT_CONFIG) -> ProxyConfig:
    """Read proxy.json if present, then let RH_PROXY_* environment variables win."""
    cfg = ProxyConfig()
    raw: Dict[str, Any] = {}
    path = Path(path)
    if path.exists():
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                raw = parsed
        except Exception as exc:
            raise ValueError(f"could not parse {path}: {exc}") from exc
    cfg.enabled = bool(raw.get("enabled", False))
    cfg.url = str(raw.get("url", "") or "")
    cfg.token = str(raw.get("token", "") or "")
    cfg.platform = _tag(raw.get("platform"), DEFAULT_PLATFORM)
    cfg.account = _tag(raw.get("account"))
    cfg.sticky = str(raw.get("sticky", "credential") or "credential").strip().lower()
    cfg.trust_headers = bool(raw.get("trust_headers", True))
    cfg.verify_tls = bool(raw.get("verify_tls", True))
    for key, value in (raw.get("bindings") or {}).items():
        cfg.bindings[str(key)] = _parse_binding(value)
    env_url = _env_str("RH_PROXY_URL")
    if env_url is not None:
        cfg.url = env_url
    env_token = _env_str("RH_PROXY_TOKEN")
    if env_token is not None:
        cfg.token = env_token
    env_platform = _env_str("RH_PROXY_PLATFORM")
    if env_platform:
        cfg.platform = _tag(env_platform, DEFAULT_PLATFORM)
    env_account = _env_str("RH_PROXY_ACCOUNT")
    if env_account is not None:
        cfg.account = _tag(env_account)
    env_sticky = _env_str("RH_PROXY_STICKY")
    if env_sticky:
        cfg.sticky = env_sticky.lower()
    for name, attr in (("RH_PROXY_ENABLED", "enabled"), ("RH_PROXY_TRUST_HEADERS", "trust_headers"), ("RH_PROXY_VERIFY_TLS", "verify_tls")):
        flag = _env_bool(name)
        if flag is not None:
            setattr(cfg, attr, flag)
    # An explicit url with no enabled flag anywhere is taken as intent to use it.
    if cfg.url and "enabled" not in raw and _env_bool("RH_PROXY_ENABLED") is None:
        cfg.enabled = True
    if cfg.sticky not in STICKY_MODES:
        raise ValueError(f"sticky must be one of {', '.join(STICKY_MODES)}, got {cfg.sticky!r}")
    cfg.url = _normalize_url(cfg.url)
    return cfg
@dataclass(frozen=True)
class ProxyRoute:
    """One resolved egress identity, ready to hand to requests."""
    platform: str
    account: str
    label: str
    proxies: Dict[str, str]
    verify: bool = True
    @property
    def sticky(self) -> bool:
        return bool(self.account)
def _build_route(cfg: ProxyConfig, platform: str, account: str) -> ProxyRoute:
    parts = urlsplit(cfg.url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port else ""
    # Resin reads the proxy username as Platform.Account; a bare platform means
    # "any node in that pool", which is the non-sticky case.
    user = f"{platform}.{account}" if account else platform
    auth = f"{quote(user, safe='')}:{quote(cfg.token, safe='')}@" if user else ""
    url = f"{parts.scheme}://{auth}{host}{port}"
    return ProxyRoute(
        platform=platform,
        account=account,
        label=f"{user}@{host}{port}",
        proxies={"http": url, "https": url},
        verify=cfg.verify_tls,
    )
class ProxyResolver:
    """Resolves each upstream call to an egress identity. Safe to share across threads."""
    def __init__(self, path: Path = DEFAULT_CONFIG, config: Optional[ProxyConfig] = None) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._config = config if config is not None else load_config(self.path)
        self._error: Optional[str] = None
    @property
    def config(self) -> ProxyConfig:
        with self._lock:
            return self._config
    @property
    def active(self) -> bool:
        return self.config.active
    def reload(self) -> bool:
        """Re-read config. A bad file is recorded and the previous config kept."""
        try:
            cfg = load_config(self.path)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
            return False
        with self._lock:
            self._config = cfg
            self._error = None
        return True
    def resolve(
        self,
        credential_id: Optional[str] = None,
        *,
        platform: Optional[str] = None,
        account: Optional[str] = None,
        client_tag: Optional[str] = None,
    ) -> Optional[ProxyRoute]:
        """None means "go direct". Precedence: request > per-credential binding > sticky mode > config default."""
        cfg = self.config
        if not cfg.active:
            return None
        binding = cfg.bindings.get(str(credential_id or ""), {})
        req_platform = _tag(platform) if cfg.trust_headers else ""
        req_account = _tag(account) if cfg.trust_headers else ""
        chosen_platform = req_platform or binding.get("platform") or cfg.platform or DEFAULT_PLATFORM
        chosen_account = req_account or binding.get("account") or self._sticky_account(cfg, credential_id, client_tag)
        return _build_route(cfg, chosen_platform, chosen_account)
    def varies_by_credential(self) -> bool:
        """Does the egress identity differ from one credential to the next?

        If it does, a dead proxy node takes out only the credentials pinned to it, so the
        caller should try another credential rather than failing the request. Per-credential
        bindings count too: they pin specific credentials to their own identities.
        """
        cfg = self.config
        if not cfg.active:
            return False
        return cfg.sticky == "credential" or bool(cfg.bindings)
    def resolve_service(self) -> Optional[ProxyRoute]:
        """Route for the app's own metadata traffic (the model catalog), which belongs to
        no credential and no caller. Keeps it off the host's bare address."""
        cfg = self.config
        if not cfg.active:
            return None
        account = "" if cfg.sticky == "none" else (cfg.account or "catalog")
        return _build_route(cfg, cfg.platform, account)
    @staticmethod
    def _sticky_account(cfg: ProxyConfig, credential_id: Optional[str], client_tag: Optional[str]) -> str:
        if cfg.sticky == "none":
            return ""
        if cfg.sticky == "credential":
            # One stable egress IP per upstream cookie: the pairing the site sees stays put.
            tag = _tag(credential_id)
            return f"cred_{tag}" if tag else cfg.account
        if cfg.sticky == "client":
            tag = _tag(client_tag)
            return tag or cfg.account
        return cfg.account  # "fixed": one egress for the whole instance
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            cfg, error = self._config, self._error
        out: Dict[str, Any] = {
            "enabled": cfg.enabled,
            "active": cfg.active,
            "sticky": cfg.sticky,
            "platform": cfg.platform,
            "trust_headers": cfg.trust_headers,
            "verify_tls": cfg.verify_tls,
            "bindings": len(cfg.bindings),
            "has_token": bool(cfg.token),
        }
        if cfg.url:
            parts = urlsplit(cfg.url)
            out["endpoint"] = f"{parts.scheme}://{parts.hostname}{f':{parts.port}' if parts.port else ''}"
        if cfg.account:
            out["default_account"] = cfg.account
        if error:
            out["config_error"] = error
        return out
def probe(route: Optional[ProxyRoute], timeout: Tuple[int, int] = CHECK_TIMEOUT) -> Dict[str, Any]:
    """Ask an echo service which IP this route actually egresses from."""
    import requests
    out: Dict[str, Any] = {"label": route.label if route else "direct", "sticky": bool(route and route.sticky)}
    if route:
        out["platform"], out["account"] = route.platform, route.account or None
    try:
        resp = requests.get(
            CHECK_URL,
            proxies=route.proxies if route else None,
            verify=route.verify if route else True,
            timeout=timeout,
        )
        resp.raise_for_status()
        out["ok"] = True
        out["ip"] = (resp.json() or {}).get("ip")
    except Exception as exc:
        out["ok"] = False
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


