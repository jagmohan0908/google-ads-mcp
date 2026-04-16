"""Tests for hosted read-only tools."""

import json

import pytest
from fastmcp.exceptions import ToolError

from ads_mcp.tools import hosted


@pytest.fixture(autouse=True)
def reset_caches():
  hosted._TOKEN_CACHE.clear()  # pylint: disable=protected-access
  hosted._RATE_LIMIT_BUCKETS.clear()  # pylint: disable=protected-access
  yield
  hosted._TOKEN_CACHE.clear()  # pylint: disable=protected-access
  hosted._RATE_LIMIT_BUCKETS.clear()  # pylint: disable=protected-access


def _configure_env(monkeypatch):
  monkeypatch.setenv(
      "MCP_LABEL_CONFIG_JSON",
      json.dumps(
          {
              "acct_ads": {
                  "provider": "google_ads",
                  "allowedTools": ["ads_query_report"],
                  "accountContext": {"customer_id": "1234567890"},
                  "credentials": {
                      "client_id": "id",
                      "client_secret": "secret",
                      "refresh_token": "refresh",
                      "developer_token": "dev",
                  },
              }
          }
      ),
  )
  monkeypatch.setenv(
      "MCP_AUTH_TOKENS_JSON", json.dumps({"analyst_1": "token-abc"})
  )
  monkeypatch.setenv(
      "MCP_LABEL_RBAC_JSON", json.dumps({"analyst_1": ["acct_ads"]})
  )


def test_get_label_registry(monkeypatch):
  _configure_env(monkeypatch)
  registry = hosted.get_label_registry()
  assert "acct_ads" in registry
  assert registry["acct_ads"].provider == "google_ads"


def test_list_labels(monkeypatch):
  _configure_env(monkeypatch)
  output = hosted.list_labels(caller_id="analyst_1", auth_token="token-abc")
  assert output == {"labels": ["acct_ads"]}


def test_list_labels_with_default_auth(monkeypatch):
  _configure_env(monkeypatch)
  monkeypatch.setenv("MCP_DEFAULT_CALLER_ID", "analyst_1")
  monkeypatch.setenv("MCP_DEFAULT_AUTH_TOKEN", "token-abc")
  output = hosted.list_labels()
  assert output == {"labels": ["acct_ads"]}


def test_list_labels_unauthorized(monkeypatch):
  _configure_env(monkeypatch)
  with pytest.raises(ToolError, match="Unauthorized caller"):
    hosted.list_labels(caller_id="analyst_1", auth_token="bad")


def test_rate_limit(monkeypatch):
  _configure_env(monkeypatch)
  monkeypatch.setenv("MCP_RATE_LIMIT_PER_MIN", "1")
  hosted._check_rate_limit("analyst_1", "acct_ads")  # pylint: disable=protected-access
  with pytest.raises(ToolError, match="Rate limit exceeded"):
    hosted._check_rate_limit("analyst_1", "acct_ads")  # pylint: disable=protected-access


def test_refresh_access_token_cached(monkeypatch):
  _configure_env(monkeypatch)
  config = hosted.get_label_registry()["acct_ads"]
  hosted._TOKEN_CACHE["acct_ads:v1"] = ("cached-token", 9999999999)  # pylint: disable=protected-access
  assert hosted._refresh_access_token(config) == "cached-token"  # pylint: disable=protected-access


def test_resolve_auth_requires_args_or_defaults():
  with pytest.raises(ToolError, match="caller_id and auth_token are required"):
    hosted._resolve_auth(None, None)  # pylint: disable=protected-access


def test_resolve_label_requires_arg_or_default():
  with pytest.raises(ToolError, match="label is required"):
    hosted._resolve_label(None)  # pylint: disable=protected-access
