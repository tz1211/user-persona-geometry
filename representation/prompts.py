from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dimensions.experiment_config import BASE_SYSTEM_PROMPT

# Qwen chat template (see tokenizer chat_template): user turn starts with this marker.
_IM_START_USER = "<|im_start|>user"


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    messages: list[dict[str, str]]
    token_ids: list[int]
    positions: dict[str, int | None]


def build_system_prompt(level_description: str | None) -> str:
    if level_description:
        return f"{BASE_SYSTEM_PROMPT}\n{level_description.strip()}"
    return BASE_SYSTEM_PROMPT


def build_messages(record: dict[str, Any], system_prompt: str) -> list[dict[str, str]]:
    if isinstance(record.get("messages"), list):
        messages = list(record["messages"])
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_prompt}, *messages]
        return messages
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(record["prompt"])},
    ]


def _last_token_ending_at_or_before(offsets: list[tuple[int, int]], char_end: int) -> int:
    candidates = [
        idx for idx, (start, end) in enumerate(offsets)
        if end <= char_end and end > start
    ]
    if not candidates:
        raise ValueError(f"Could not resolve token ending before char offset {char_end}")
    return candidates[-1]


def _token_index_for_char_end(tokenizer: Any, rendered: str, char_end: int) -> int:
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is not None:
        return _last_token_ending_at_or_before(offsets, char_end)

    # Fallback for slow tokenizers. The boundary is at a message-content end, so
    # prefix tokenization is stable enough for this extraction target.
    prefix_ids = tokenizer(rendered[:char_end], add_special_tokens=False)["input_ids"]
    if not prefix_ids:
        raise ValueError(f"Could not resolve token before char offset {char_end}")
    return len(prefix_ids) - 1


def _find_required(haystack: str, needle: str, start: int = 0, label: str = "substring") -> int:
    idx = haystack.find(needle, start)
    if idx < 0:
        raise ValueError(f"Could not locate {label} in rendered chat template")
    return idx


def render_with_positions(
    *,
    tokenizer: Any,
    record: dict[str, Any],
    condition: dict[str, Any],
) -> RenderedPrompt:
    """Render a chat prompt and resolve Section 6.1 token positions P1-P4.

    P1: last token overlapping end of inner system ``messages[0]["content"]``.
    P2: last token overlapping end of ``user`` role label after ``<|im_start|>``
        (boundary before user message body; ChatML analogue of P4 at assistant).
    """
    system_prompt = build_system_prompt(condition.get("level_description"))
    messages = build_messages(record, system_prompt)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError("Rendered prompt tokenized to an empty sequence")

    system_start = _find_required(rendered, system_prompt, label="system prompt")
    system_end = system_start + len(system_prompt)

    marker_idx = rendered.find(_IM_START_USER)
    if marker_idx < 0:
        raise ValueError(
            f"No {_IM_START_USER!r} marker in rendered prompt for condition {condition['id']!r}. "
            "Update _IM_START_USER if the tokenizer chat_template format changes.",
        )
    user_role_end = marker_idx + len(_IM_START_USER)

    positions: dict[str, int | None] = {
        "P1": _token_index_for_char_end(tokenizer, rendered, system_end),
        "P2": _token_index_for_char_end(tokenizer, rendered, user_role_end),
        "P3": None,
        "P4": len(token_ids) - 1,
    }

    user_message = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), None)
    if user_message is None:
        raise ValueError("Rendered prompt has no user message")
    user_start = _find_required(rendered, user_message, system_end, "user message")
    positions["P3"] = _token_index_for_char_end(
        tokenizer,
        rendered,
        user_start + len(user_message),
    )

    if not (positions["P1"] < positions["P2"] < positions["P3"] < positions["P4"]):
        raise ValueError(f"Unexpected position ordering for {condition['id']}: {positions}")

    return RenderedPrompt(
        text=rendered,
        messages=messages,
        token_ids=token_ids,
        positions=positions,
    )

