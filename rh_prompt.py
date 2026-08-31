"""Fit a prompt inside the upstream's per-request ceiling.

tryingopen.com rejects oversized submissions with HTTP 413 ("There's too much text
in this chat now"), a limit well below the models' advertised context. Shaping happens
in three escalating steps, cheapest first:

1. squeeze  - lossless whitespace collapse
2. elide    - cut the middle out of huge messages, keeping head and tail
3. drop     - discard the oldest turns, always keeping the first message

The first message is never dropped: to_ui_messages() folds the system instructions
into it, so losing it would silently discard the system prompt.

The tool-calling spec is deliberately NOT part of what gets shaped. It is appended by
the caller after shaping, against reserve_tokens held back here, because a spec that
gets elided mid-JSON turns tool calling off without saying so.
"""
from __future__ import annotations
import os
import re
from typing import Any, Dict, List, Optional, Tuple
# CJK ranges plus kana and hangul: roughly one token per character, unlike latin text.
_CJK = re.compile(r"[㐀-䶿一-鿿぀-ヿ가-힯豈-﫿　-〿＀-￯]")
_SPACE_RUNS = re.compile(r"[ \t]{3,}")
_BLANK_RUNS = re.compile(r"\n\s*\n(\s*\n)+")
_TRAILING_WS = re.compile(r"[ \t]+(?=\n)")
DEFAULT_TOKEN_BUDGET = 16000
MIN_TOKEN_BUDGET = 1200
SHRINK_FACTOR = 0.55
FIRST_MESSAGE_SHARE = 0.40
MIN_MESSAGE_TOKENS = 120
ELIDE_MARK = "\n\n[... {n} characters elided from the middle ...]\n\n"
DROP_MARK = "[... {n} earlier message(s) omitted to fit the upstream limit ...]"
class PromptTooLargeError(RuntimeError):
    """Upstream refused the prompt as too large and shaping could not get it under."""
    def __init__(self, message: str, report: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.report = report or {}
def token_budget_default() -> int:
    """Env-tunable so the ceiling can be retuned without a rebuild."""
    raw = (os.environ.get("RH_PROMPT_TOKEN_BUDGET") or "").strip()
    if raw:
        try:
            return max(MIN_TOKEN_BUDGET, int(raw))
        except ValueError:
            pass
    return DEFAULT_TOKEN_BUDGET
def estimate_tokens(text: str) -> int:
    """Cheap CJK-aware estimate. Chinese runs ~1 token/char where latin runs ~4 chars/token,
    so a plain len()/4 underestimates CJK prompts by several times."""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk + 3) // 4
def part_tokens(part: Dict[str, Any]) -> int:
    if part.get("type") == "text":
        return estimate_tokens(str(part.get("text") or ""))
    # Attachments cost their wire length; a base64 data URI is charged like any payload.
    return (len(str(part.get("url") or "")) + 3) // 4
def message_tokens(msg: Dict[str, Any]) -> int:
    return sum(part_tokens(p) for p in msg.get("parts") or [])
def total_tokens(msgs: List[Dict[str, Any]]) -> int:
    return sum(message_tokens(m) for m in msgs)
def attachment_tokens(msgs: List[Dict[str, Any]]) -> int:
    return sum(part_tokens(p) for m in msgs for p in (m.get("parts") or []) if p.get("type") != "text")
def squeeze(text: str) -> str:
    """Lossless: collapse indent runs, blank-line pileups and trailing spaces."""
    if not text:
        return text
    out = _TRAILING_WS.sub("", text)
    out = _SPACE_RUNS.sub("  ", out)
    out = _BLANK_RUNS.sub("\n\n", out)
    return out.strip("\n")
def squeeze_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    parts = []
    for p in msg.get("parts") or []:
        if p.get("type") == "text":
            squeezed = squeeze(str(p.get("text") or ""))
            if squeezed:
                parts.append({**p, "text": squeezed})
        else:
            parts.append(p)
    return {**msg, "parts": parts}
def elide_middle(text: str, target_tokens: int) -> Tuple[str, int]:
    """Keep the head and the tail, cut the middle. Returns (text, characters removed).

    Head-and-tail beats head-only: the end of a log, diff or tool result usually carries
    the conclusion, and that is what a tail-truncating trim throws away first.
    """
    if target_tokens <= 0 or not text:
        return "", len(text)
    if estimate_tokens(text) <= target_tokens:
        return text, 0
    # Convert the token target back to characters at this text's own density.
    density = max(1.0, len(text) / max(1, estimate_tokens(text)))
    keep_chars = max(200, int(target_tokens * density) - len(ELIDE_MARK) - 24)
    if keep_chars >= len(text):
        return text, 0
    head = keep_chars * 2 // 3
    tail = keep_chars - head
    removed = len(text) - head - tail
    kept_head = text[:head].rstrip()
    kept_tail = text[len(text) - tail:].lstrip() if tail else ""
    return kept_head + ELIDE_MARK.format(n=removed) + kept_tail, removed
def fit_message(msg: Dict[str, Any], target_tokens: int) -> Tuple[Dict[str, Any], int]:
    """Shrink one message's text parts to roughly target_tokens. Attachments are left
    alone: they cannot be partially sent, and silently dropping an image is worse than
    reporting that it does not fit."""
    fixed = attachment_tokens([msg])
    text_budget = max(0, target_tokens - fixed)
    text_parts = [p for p in msg.get("parts") or [] if p.get("type") == "text"]
    if not text_parts:
        return msg, 0
    current = sum(part_tokens(p) for p in text_parts)
    if current <= text_budget:
        return msg, 0
    removed_total = 0
    parts: List[Dict[str, Any]] = []
    remaining = text_budget
    # Spend the budget on the last text part first: it is the most recent thing said.
    share = {id(p): 0 for p in text_parts}
    for p in reversed(text_parts):
        want = min(part_tokens(p), max(MIN_MESSAGE_TOKENS // 2, remaining))
        share[id(p)] = want
        remaining = max(0, remaining - want)
    for p in msg.get("parts") or []:
        if p.get("type") != "text":
            parts.append(p)
            continue
        new_text, removed = elide_middle(str(p.get("text") or ""), share.get(id(p), 0))
        removed_total += removed
        if new_text:
            parts.append({**p, "text": new_text})
    return {**msg, "parts": parts}, removed_total
def _with_drop_note(msgs: List[Dict[str, Any]], report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Tell the model that history was removed, so it does not treat the gap as the
    whole conversation."""
    n = report.get("dropped_messages") or 0
    if n <= 0 or not msgs:
        return msgs
    role = msgs[1]["role"] if len(msgs) > 1 else msgs[0]["role"]
    note = {"id": "trim-note", "role": role, "parts": [{"type": "text", "text": DROP_MARK.format(n=n)}]}
    return [msgs[0], note] + list(msgs[1:])
def shape(
    msgs: List[Dict[str, Any]],
    token_budget: Optional[int] = None,
    max_messages: Optional[int] = None,
    reserve_tokens: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return (messages, report) with the prompt shaped to fit token_budget.

    reserve_tokens holds back part of the budget for text the caller appends after
    shaping - the tool-calling spec. Without the reserve, appending afterwards would
    push the request back over the ceiling that shaping just brought it under.
    """
    budget = max(MIN_TOKEN_BUDGET, token_budget or token_budget_default())
    if reserve_tokens > 0:
        # Never let the reserve eat the whole budget: the conversation still needs room.
        budget = max(MIN_TOKEN_BUDGET // 2, budget - reserve_tokens)
    before = total_tokens(msgs)
    report: Dict[str, Any] = {
        "budget_tokens": budget, "before_tokens": before, "after_tokens": before,
        "squeezed": False, "dropped_messages": 0, "elided_chars": 0,
        "attachment_tokens": attachment_tokens(msgs), "shaped": False,
    }
    if not msgs:
        return msgs, report
    work = msgs
    # Turn-count cap, applied whatever the token total: keep the first message (it holds
    # the folded system prompt) plus the newest turns.
    if max_messages and len(work) > max_messages:
        keep_tail = max(1, max_messages - 1)
        report["dropped_messages"] = len(work) - 1 - keep_tail
        work = [work[0]] + work[-keep_tail:]
        report["shaped"] = True
    if total_tokens(work) <= budget:
        report["after_tokens"] = total_tokens(work)
        return (_with_drop_note(work, report) if report["dropped_messages"] else work), report
    # Step 1: lossless squeeze.
    work = [squeeze_message(m) for m in work]
    work = [m for m in work if m.get("parts")]
    report["squeezed"] = True
    report["shaped"] = True
    if total_tokens(work) <= budget or not work:
        report["after_tokens"] = total_tokens(work)
        return (_with_drop_note(work, report) if report["dropped_messages"] else work), report
    # Step 2: the first message carries the folded system prompt, so it is kept and
    # capped rather than dropped.
    first, rest = work[0], work[1:]
    first_cap = max(MIN_MESSAGE_TOKENS, int(budget * FIRST_MESSAGE_SHARE))
    if message_tokens(first) > first_cap:
        first, removed = fit_message(first, first_cap)
        report["elided_chars"] += removed
    remaining = budget - message_tokens(first)
    # Step 3: walk backwards from the newest turn, keeping whatever fits.
    kept: List[Dict[str, Any]] = []
    for msg in reversed(rest):
        cost = message_tokens(msg)
        if cost <= remaining:
            kept.append(msg)
            remaining -= cost
            continue
        if remaining >= MIN_MESSAGE_TOKENS:
            # Spend what is left rather than stopping with budget unused: an elided turn
            # carries more than a dropped one.
            shrunk, removed = fit_message(msg, remaining)
            report["elided_chars"] += removed
            kept.append(shrunk)
            remaining -= message_tokens(shrunk)
        break
    kept.reverse()
    report["dropped_messages"] += len(rest) - len(kept)
    out = _with_drop_note([first] + kept, report)
    out = _enforce_budget(out, budget, report)
    report["after_tokens"] = total_tokens(out)
    return out, report
def _enforce_budget(msgs: List[Dict[str, Any]], budget: int, report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Make the budget a guarantee rather than an aim.

    The elide markers and per-message rounding each add a few tokens back, so the
    step-by-step passes can land slightly over. Rather than chase every source of
    slack, trim the largest message until the total actually fits.
    """
    for _ in range(8):
        total = total_tokens(msgs)
        if total <= budget or not msgs:
            return msgs
        over = total - budget
        idx = max(range(len(msgs)), key=lambda i: message_tokens(msgs[i]))
        target = message_tokens(msgs[idx]) - over - 16
        if target < MIN_MESSAGE_TOKENS // 2:
            # Largest message cannot absorb it; drop the oldest turn after the first.
            if len(msgs) > 2:
                report["dropped_messages"] = report.get("dropped_messages", 0) + 1
                msgs = [msgs[0]] + msgs[2:]
                continue
            target = max(1, target)
        shrunk, removed = fit_message(msgs[idx], target)
        report["elided_chars"] = report.get("elided_chars", 0) + removed
        if removed == 0:
            return msgs  # nothing left to give (attachments only)
        msgs = msgs[:idx] + [shrunk] + msgs[idx + 1:]
    return msgs
class LearnedLimit:
    """Remembers the size the upstream actually refused, so later requests pre-trim
    instead of spending another rejected round trip to rediscover the same ceiling."""
    def __init__(self, start: Optional[int] = None) -> None:
        self.start = max(MIN_TOKEN_BUDGET, start or token_budget_default())
        self.known_bad: Optional[int] = None
    @property
    def budget(self) -> int:
        if self.known_bad is None:
            return self.start
        return max(MIN_TOKEN_BUDGET, min(self.start, int(self.known_bad * SHRINK_FACTOR)))
    def record_rejection(self, tokens: int) -> None:
        if tokens <= 0:
            return
        self.known_bad = tokens if self.known_bad is None else min(self.known_bad, tokens)
    def reset(self) -> Dict[str, Any]:
        """Forget the learned ceiling and re-read the env budget.

        A rejection is remembered for the process lifetime, so one oversized request
        permanently shrank the budget for every later one - and a smaller budget squeezes
        the tool spec, which is what turns tool calling off. Clearing it needed a restart
        until this was reachable from /admin/reload.
        """
        before = self.budget
        self.known_bad = None
        self.start = token_budget_default()
        return {"budget_before": before, "budget_after": self.budget}
    @staticmethod
    def shrink(tokens: int) -> int:
        return max(MIN_TOKEN_BUDGET, int(tokens * SHRINK_FACTOR))
    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"budget_tokens": self.budget, "start_tokens": self.start}
        if self.known_bad is not None:
            out["rejected_at_tokens"] = self.known_bad
        return out


