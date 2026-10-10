"""Load the one supported local credential without evaluating dotenv syntax."""
from __future__ import annotations

import os
import re
from pathlib import Path

_KEY = "ANTHROPIC_API_KEY"
_INVALID = "Invalid ANTHROPIC_API_KEY entry in .env; use one single-line KEY=value assignment."


def _value(text: str) -> str:
    text = text.strip()
    if text.startswith(("'", '"')):
        end = text.find(text[0], 1)
        if end < 0:
            raise ValueError(_INVALID)
        value, trailing = text[1:end], text[end + 1:].strip()
        if trailing and not trailing.startswith("#"):
            raise ValueError(_INVALID)
    else:
        value = re.split(r"[ \t]+#", text, maxsplit=1)[0].rstrip()
    if not value or (not text.startswith(("'", '"')) and value.startswith("#")):
        raise ValueError(_INVALID)
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(_INVALID)
    return value


def load_anthropic_key(path: Path) -> bool:
    """Load only ANTHROPIC_API_KEY, preserving even an empty exported value.

    Supports comments, optional ``export``, and literal single/double quotes.
    No interpolation, escape decoding, shell execution, or other variables are
    supported. Errors never include file contents or credential values.
    """
    if _KEY in os.environ:
        return False
    try:
        content = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return False
    except (OSError, UnicodeError):
        raise ValueError("Cannot read local .env credential file.") from None
    candidate: str | None = None
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^export[ \t]+", "", line)
        if not re.match(r"ANTHROPIC_API_KEY(?=$|[\s=])", line):
            continue
        match = re.fullmatch(r"ANTHROPIC_API_KEY[ \t]*=(.*)", line)
        if match is None or candidate is not None:
            raise ValueError(_INVALID)
        candidate = _value(match.group(1))
    if candidate is None:
        return False
    # Preserve a value another caller might have exported while the file was read.
    return os.environ.setdefault(_KEY, candidate) == candidate
