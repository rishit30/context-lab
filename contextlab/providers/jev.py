"""Small, no-retry transport for TypeSafe's verified System One HTTP API.

The origin is fixed deliberately: a credential must never follow a redirect or
an untrusted API-base environment variable. No inference happens on import.
"""
from __future__ import annotations

import http.client
import json
import math
import os
import ssl
from typing import Any

from contextlab.core.models import Usage

HOST = "api.typesafe.ai"
MAX_REQUEST_BYTES = 32000
MAX_RESPONSE_BYTES = 512000


class JevError(RuntimeError):
    """Sanitized error: no server body, request contents or credential attached."""


class JevClient:
    def __init__(self, api_key: str, *, timeout_seconds: float = 10) -> None:
        if not isinstance(api_key, str) or not api_key.strip() or any(c in api_key for c in "\r\n"):
            raise JevError("TYPESAFE_API_KEY is missing or invalid")
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 30:
            raise JevError("Jev timeout must be between 0 and 30 seconds")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls, *, timeout_seconds: float = 10) -> JevClient:
        return cls(os.environ.get("TYPESAFE_API_KEY", ""), timeout_seconds=timeout_seconds)

    def _request(self, method: str, path: str, body: bytes | None = None) -> dict[str, Any]:
        connection = http.client.HTTPSConnection(
            HOST, timeout=self.timeout_seconds, context=ssl.create_default_context(),
        )
        try:
            connection.request(method, path, body=body, headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json", "Accept": "application/json",
            })
            response = connection.getresponse()
            # http.client does not follow redirects or retry failed requests.
            if response.status != 200:
                raise JevError(f"TypeSafe HTTP status {response.status}")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise JevError("TypeSafe response exceeded the local size limit")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise JevError("TypeSafe response must be a JSON object")
            return result
        except JevError:
            raise
        except Exception:
            raise JevError("TypeSafe request failed; provider usage may be unknown") from None
        finally:
            connection.close()

    def list_models(self) -> dict[str, Any]:
        """Authenticated GET; no model inference or tool contents sent."""
        result = self._request("GET", "/v1/models")
        rows = result.get("models")
        if not isinstance(rows, list) or not all(
            isinstance(row, dict) and isinstance(row.get("name"), str) for row in rows
        ):
            raise JevError("TypeSafe returned an invalid model list")
        return {"models": [{key: row[key] for key in ("name", "description", "release_date")
                            if isinstance(row.get(key), str)} for row in rows]}

    @staticmethod
    def encode_request(payload: dict[str, Any]) -> bytes:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        if len(body) > MAX_REQUEST_BYTES:
            raise JevError("TypeSafe request exceeded the local size limit")
        return body

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/v1/systemone", self.encode_request(payload))


def normalize_jev_usage(receipt: dict[str, Any]) -> Usage:
    raw = receipt.get("usage")
    if not isinstance(raw, dict):
        raise JevError("TypeSafe usage receipt is missing")
    for name in ("input_tokens", "output_tokens"):
        value = raw.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise JevError("TypeSafe usage receipt is invalid")
    return Usage(input_tokens=raw["input_tokens"], output_tokens=raw["output_tokens"])


def validate_noul_answers(receipt: dict[str, Any], question_ids: set[str], model: str) -> dict[str, float]:
    if receipt.get("model") != model:
        raise JevError("TypeSafe resolved model does not match the pinned model")
    answers = receipt.get("answers")
    if not isinstance(answers, dict) or set(answers) != question_ids:
        raise JevError("TypeSafe answer IDs do not match the request")
    probabilities = {}
    for question_id, answer in answers.items():
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise JevError("TypeSafe answer type must be noul")
        value = answer.get("noul")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise JevError("TypeSafe probability must be finite and between zero and one")
        probabilities[question_id] = float(value)
    return probabilities
