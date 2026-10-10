from __future__ import annotations

import json

import pytest

from contextlab.providers.jev import JevClient, JevError, normalize_jev_usage, validate_noul_answers


def receipt(value=0.1):
    return {"model": "jev-1.13.0", "answers": {"c0": {"type": "noul", "noul": value}},
            "usage": {"input_tokens": 123, "output_tokens": 1}}


def test_valid_response_does_not_confuse_noul_with_confidence():
    assert validate_noul_answers(receipt(), {"c0"}, "jev-1.13.0") == {"c0": 0.1}
    assert normalize_jev_usage(receipt()).input_tokens == 123


@pytest.mark.parametrize("value", [None, True, "0.1", -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probabilities_fail_closed(value):
    with pytest.raises(JevError):
        validate_noul_answers(receipt(value), {"c0"}, "jev-1.13.0")


@pytest.mark.parametrize("change", [
    {"model": "another-model"}, {"answers": {}},
    {"answers": {"c0": {"type": "choice", "choice": "drop"}}},
    {"answers": {"c0": {"type": "noul", "noul": 0.1}, "extra": {"type": "noul", "noul": 0}}},
])
def test_wrong_model_missing_or_unexpected_answers_rejected(change):
    with pytest.raises(JevError):
        validate_noul_answers({**receipt(), **change}, {"c0"}, "jev-1.13.0")


@pytest.mark.parametrize("value", [None, True, "123", -1, 1.5])
def test_usage_requires_real_nonnegative_integer_counts(value):
    with pytest.raises(JevError):
        normalize_jev_usage({"usage": {"input_tokens": value, "output_tokens": 0}})


def test_request_size_rejects_before_connection(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Oversized request reached transport")
    monkeypatch.setattr("http.client.HTTPSConnection", unexpected)
    with pytest.raises(JevError, match="size limit"):
        JevClient("test-not-a-key").infer({"state": "x" * 32001})


@pytest.mark.parametrize("status", [301, 401, 429, 503])
def test_transport_never_redirects_retries_or_discloses_response(monkeypatch, status):
    sent = []
    class Response:
        def read(self, _):
            pytest.fail("Error response bodies must not be logged or parsed")
    class Connection:
        def __init__(self, host, **kwargs):
            assert host == "api.typesafe.ai"
        def request(self, method, path, **kwargs):
            sent.append((method, path))
        def getresponse(self):
            response = Response()
            response.status = status
            return response
        def close(self):
            pass
    monkeypatch.setattr("http.client.HTTPSConnection", Connection)
    with pytest.raises(JevError, match=f"status {status}") as error:
        JevClient("test-not-a-key").list_models()
    assert "test-not-a-key" not in str(error.value)
    assert sent == [("GET", "/v1/models")]


def test_transport_models_get_and_inference_post(monkeypatch):
    sent = []
    class Response:
        status = 200
        def read(self, _):
            return json.dumps({"models": [{"name": "jev-latest", "description": "test"}]}).encode()
    class Connection:
        def __init__(self, host, **kwargs):
            pass
        def request(self, method, path, **kwargs):
            assert kwargs["headers"]["Authorization"] == "Bearer test-not-a-key"
            sent.append((method, path, kwargs.get("body")))
        def getresponse(self):
            return Response()
        def close(self):
            pass
    monkeypatch.setattr("http.client.HTTPSConnection", Connection)
    client = JevClient("test-not-a-key")
    assert client.list_models()["models"][0]["name"] == "jev-latest"
    client.infer({"model": "jev-1.13.0", "state": "example", "questions": {}})
    assert sent[0] == ("GET", "/v1/models", None)
    assert sent[1][:2] == ("POST", "/v1/systemone")


def test_transport_error_is_sanitized(monkeypatch):
    class Connection:
        def __init__(self, *args, **kwargs):
            pass
        def request(self, *args, **kwargs):
            raise OSError("sensitive server request detail")
        def close(self):
            pass
    monkeypatch.setattr("http.client.HTTPSConnection", Connection)
    with pytest.raises(JevError) as error:
        JevClient("test-not-a-key").list_models()
    assert "sensitive" not in str(error.value)
    assert error.value.__suppress_context__
