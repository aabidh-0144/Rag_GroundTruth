"""Ask the judge LLM a question and get back validated, typed data.

Local models wrap JSON in prose or code fences, drop fields, or invent values
outside a schema. `judge_json` handles all of that in one place:

1. call the client in JSON mode
2. pull the JSON object out of the reply, validate it against a pydantic model
3. on failure, send ONE repair request quoting the validation error
4. still invalid -> raise JudgeError (the metric is then recorded as "failed")

It never returns a default or partially-filled verdict: a judge that could not
answer has not scored anything.
"""

from __future__ import annotations

import json
import re
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from groundtruth.evaluation.prompts import REPAIR_USER
from groundtruth.llm import LLMClient

T = TypeVar("T", bound=BaseModel)

# Extra validation a schema alone can't express (e.g. "one verdict per block").
# Raises ValueError to reject the reply, which triggers the repair turn.
Check = Callable[[BaseModel], None]

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class JudgeError(RuntimeError):
    """The judge replied, but not with data matching the requested schema."""


def extract_json_object(reply: str) -> str:
    """Best-effort: the outermost {...} in the reply, minus any code fence."""
    text = _FENCE_RE.sub("", reply.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return text[start : end + 1]


def _parse(reply: str, model: type[T], check: Check | None) -> T:
    try:
        parsed = model.model_validate(json.loads(extract_json_object(reply)))
        if check is not None:
            check(parsed)
        return parsed
    except (ValueError, ValidationError) as exc:
        # json.JSONDecodeError and pydantic's ValidationError are both ValueErrors.
        raise JudgeError(str(exc)) from exc


def judge_json(
    client: LLMClient,
    system: str,
    user: str,
    model: type[T],
    check: Check | None = None,
) -> T:
    """Return the judge's reply as an instance of `model`, or raise JudgeError."""
    reply = client.chat(system, user, json_mode=True)
    try:
        return _parse(reply, model, check)
    except JudgeError as first_error:
        repair = REPAIR_USER.format(error=str(first_error)[:1500], reply=reply[:4000])
        # The repair turn repeats the original task so the judge still has the
        # contexts in front of it, instead of "fixing" JSON from memory.
        second = client.chat(system, f"{user}\n\n{repair}", json_mode=True)
        try:
            return _parse(second, model, check)
        except JudgeError as second_error:
            raise JudgeError(
                f"judge reply invalid after one repair attempt: {second_error}"
            ) from second_error
