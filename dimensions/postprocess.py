"""Response post-processing utilities (think-tag stripping, etc.)."""

from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


def strip_think_tags(text: str) -> tuple[str, str | None]:
    """Remove ``<think>...</think>`` blocks (e.g. Qwen3 reasoning traces).

    Returns
    -------
    (cleaned_response, think_content_or_none)
    """
    match = _THINK_RE.search(text)
    if match is None:
        return text.strip(), None
    think_content = match.group(1).strip()
    cleaned = text[: match.start()] + text[match.end() :]
    return cleaned.strip(), think_content or None
