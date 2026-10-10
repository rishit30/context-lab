"""Load explicitly supported local credentials without evaluating dotenv syntax."""
from __future__ import annotations

import os
import re
from pathlib import Path

_SUPPORTED_KEYS = frozenset({"ANTHROPIC_API_KEY", "TYPESAFE_API_KEY"})


def _value(text: str, invalid: str) -> str:
    text = text.strip()
    if text.startswith(("'", '"')):
        end = text.find(text[0], 1)
        if end < 0:
            raise ValueError(invalid)
        value, trailing = text[1:end], text[end + 1:].strip()
        if trailing and not trailing.startswith("#"):
            raise ValueError(invalid)
    else:
        value = re.split(r"[ \t]+#", text, maxsplit=1)[0].rstrip()
    if not value or (not text.startswith(("'", '"')) and value.startswith("#")):
        raise ValueError(invalid)
    if any(character in value for character in "\x00\r\n"):
        raise ValueError(invalid)
    return value


def _load_key(path: Path, key: str) -> bool:
    if key not in _SUPPORTED_KEYS:
        raise ValueError("Unsupported local credential name.")
    invalid = f"Invalid {key} entry in .env; use one single-line KEY=value assignment."
    if key in os.environ:
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
        if not re.match(re.escape(key) + r"(?=$|[\s=])", line):
            continue
        match = re.fullmatch(re.escape(key) + r"[ \t]*=(.*)", line)
        if match is None or candidate is not None:
            raise ValueError(invalid)
        candidate = _value(match.group(1), invalid)
    if candidate is None:
        return False
    # Preserve a value another caller might have exported while the file was read.
    return os.environ.setdefault(key, candidate) == candidate


def load_anthropic_key(path: Path) -> bool:
    """Load only ANTHROPIC_API_KEY, preserving even an empty exported value.

    Supports comments, optional ``export``, and literal single/double quotes.
    No interpolation, escape decoding, shell execution, or other variables are
    supported. Errors never include file contents or credential values.
    """
    return _load_key(path, "ANTHROPIC_API_KEY")


def load_typesafe_key(path: Path) -> bool:
    """Load only TYPESAFE_API_KEY with the same literal-only credential rules.

    Existing exported values, including empty ones, take precedence. Anthropic
    and all other assignments are ignored, even when they are malformed.
    """
    return _load_key(path, "TYPESAFE_API_KEY")
