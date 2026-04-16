"""Hosted, label-based read-only tools for Ads, GA4, and YouTube."""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import httpx
from fastmcp.exceptions import ToolError

from ads_mcp.coordinator import mcp_server as mcp

LOGGER = logging.getLogger(__name__)

_TOKEN_CACHE: dict[str, tuple[str, float]] = {}
_RATE_LIMIT_BUCKETS: dict[str, list[float]] = defaultdict(list)


@dataclass(frozen=True)
class CredentialBundle:
  client_id: str
  client_secret: str
  refresh_token: str
  developer_token: str | None = None
  login_customer_id: str | None = None
  version: str = "v1"


@dataclass(frozen=True)
class LabelConfig:
  label: str
  provider: str
  account_context: dict[str, Any] = field(default_factory=dict)
  allowed_tools: set[str] = field(default_factory=set)
  credentials: CredentialBundle | None = None


def _load_json_env(env_key: str) -> dict[str, Any]:
  payload = os.getenv(env_key, "").strip()
  if not payload:
    return {}
  try:
    value = json.loads(payload)
  except json.JSONDecodeError as exc:
    raise ToolError(f"{env_key} is not valid JSON.") from exc
  if not isinstance(value, dict):
    raise ToolError(f"{env_key} must be a JSON object.")
  return value


def _parse_label_config(label: str, raw: dict[str, Any]) -> LabelConfig:
  provider = raw.get("provider")
  if provider not in {"google_ads", "ga4", "youtube"}:
    raise ToolError(
        f"Label {label!r} has unsupported provider: {provider!r}."
    )
  allowed_tools = raw.get("allowedTools", [])
  if not isinstance(allowed_tools, list):
    raise ToolError(f"Label {label!r} field allowedTools must be a list.")
  account_context = raw.get("accountContext", {})
  if not isinstance(account_context, dict):
    raise ToolError(f"Label {label!r} field accountContext must be an object.")

  creds_raw = raw.get("credentials")
  creds = None
  if creds_raw:
    required = {"client_id", "client_secret", "refresh_token"}
    missing = required - set(creds_raw.keys())
    if missing:
      raise ToolError(
          f"Label {label!r} credentials missing required fields: "
          f"{', '.join(sorted(missing))}."
      )
    creds = CredentialBundle(
        client_id=creds_raw["client_id"],
        client_secret=creds_raw["client_secret"],
        refresh_token=creds_raw["refresh_token"],
        developer_token=creds_raw.get("developer_token"),
        login_customer_id=creds_raw.get("login_customer_id"),
        version=creds_raw.get("version", "v1"),
    )
  return LabelConfig(
      label=label,
      provider=provider,
      account_context=account_context,
      allowed_tools=set(allowed_tools),
      credentials=creds,
  )


def get_label_registry() -> dict[str, LabelConfig]:
  """Loads and validates label config from MCP_LABEL_CONFIG_JSON."""
  raw_registry = _load_json_env("MCP_LABEL_CONFIG_JSON")
  return {label: _parse_label_config(label, raw) for label, raw in raw_registry.items()}


def _authorize_caller(caller_id: str, auth_token: str, label: str) -> None:
  if not caller_id or not auth_token:
    raise ToolError("caller_id and auth_token are required.")
  tokens = _load_json_env("MCP_AUTH_TOKENS_JSON")
  expected_token = tokens.get(caller_id)
  if not expected_token or expected_token != auth_token:
    raise ToolError("Unauthorized caller.")

  label_rbac = _load_json_env("MCP_LABEL_RBAC_JSON")
  allowed_labels = label_rbac.get(caller_id, [])
  if label not in allowed_labels:
    raise ToolError(f"Caller {caller_id!r} is not allowed to use label {label!r}.")


def _check_rate_limit(caller_id: str, label: str) -> None:
  limit = int(os.getenv("MCP_RATE_LIMIT_PER_MIN", "60"))
  bucket_key = f"{caller_id}:{label}"
  now = time.time()
  one_minute_ago = now - 60
  recent_hits = [x for x in _RATE_LIMIT_BUCKETS[bucket_key] if x >= one_minute_ago]
  if len(recent_hits) >= limit:
    raise ToolError("Rate limit exceeded. Try again in a minute.")
  recent_hits.append(now)
  _RATE_LIMIT_BUCKETS[bucket_key] = recent_hits


def _audit_log(
    caller_id: str, label: str, tool_name: str, status: str, detail: str = ""
) -> None:
  LOGGER.info(
      "audit tool=%s caller=%s label=%s status=%s detail=%s",
      tool_name,
      caller_id,
      label,
      status,
      detail,
  )


def _require_tool_access(config: LabelConfig, tool_name: str) -> None:
  if config.allowed_tools and tool_name not in config.allowed_tools:
    raise ToolError(f"Tool {tool_name!r} is not allowed for label {config.label!r}.")


def _refresh_access_token(config: LabelConfig) -> str:
  if not config.credentials:
    raise ToolError(f"Missing credentials for label {config.label!r}.")

  cache_key = f"{config.label}:{config.credentials.version}"
  cached = _TOKEN_CACHE.get(cache_key)
  now = time.time()
  if cached and cached[1] > now + 30:
    return cached[0]

  resp = httpx.post(
      "https://oauth2.googleapis.com/token",
      data={
          "client_id": config.credentials.client_id,
          "client_secret": config.credentials.client_secret,
          "refresh_token": config.credentials.refresh_token,
          "grant_type": "refresh_token",
      },
      timeout=20.0,
  )
  if resp.status_code != 200:
    raise ToolError("Failed to refresh Google access token.")
  body = resp.json()
  token = body.get("access_token")
  expires_in = int(body.get("expires_in", 3600))
  if not token:
    raise ToolError("Token refresh response did not contain access_token.")
  _TOKEN_CACHE[cache_key] = (token, now + expires_in)
  return token


def _validated_context(
    *,
    label: str,
    caller_id: str,
    auth_token: str,
    tool_name: str,
    expected_provider: str,
) -> LabelConfig:
  registry = get_label_registry()
  if label not in registry:
    raise ToolError(f"Unknown label: {label!r}.")
  config = registry[label]
  if config.provider != expected_provider:
    raise ToolError(
        f"Label {label!r} provider mismatch. Expected {expected_provider!r}."
    )
  _authorize_caller(caller_id, auth_token, label)
  _check_rate_limit(caller_id, label)
  _require_tool_access(config, tool_name)
  return config


def _resolve_auth(
    caller_id: str | None,
    auth_token: str | None,
) -> tuple[str, str]:
  """Resolves caller auth from args first, then MCP_DEFAULT_* env vars."""
  resolved_caller = caller_id or os.getenv("MCP_DEFAULT_CALLER_ID", "")
  resolved_token = auth_token or os.getenv("MCP_DEFAULT_AUTH_TOKEN", "")
  if not resolved_caller or not resolved_token:
    raise ToolError(
        "caller_id and auth_token are required. "
        "Pass arguments or set MCP_DEFAULT_CALLER_ID/MCP_DEFAULT_AUTH_TOKEN."
    )
  return resolved_caller, resolved_token


def _resolve_label(label: str | None) -> str:
  """Resolves label from args first, then MCP_DEFAULT_LABEL env var."""
  resolved_label = label or os.getenv("MCP_DEFAULT_LABEL", "")
  if not resolved_label:
    raise ToolError(
        "label is required. Pass argument or set MCP_DEFAULT_LABEL."
    )
  return resolved_label


@mcp.tool()
def list_labels(
    caller_id: str | None = None,
    auth_token: str | None = None,
) -> dict[str, list[str]]:
  """Lists labels available to the authenticated caller."""
  caller_id, auth_token = _resolve_auth(caller_id, auth_token)
  tokens = _load_json_env("MCP_AUTH_TOKENS_JSON")
  expected_token = tokens.get(caller_id)
  if not expected_token or expected_token != auth_token:
    raise ToolError("Unauthorized caller.")

  registry = get_label_registry()
  rbac = _load_json_env("MCP_LABEL_RBAC_JSON")
  allowed = set(rbac.get(caller_id, []))
  labels = sorted([x for x in registry if x in allowed])
  _audit_log(caller_id, "-", "list_labels", "success", f"count={len(labels)}")
  return {"labels": labels}


@mcp.tool()
def ads_query_report(
    query: str,
    label: str | None = None,
    caller_id: str | None = None,
    auth_token: str | None = None,
    customer_id: str | None = None,
    login_customer_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
  """Runs a read-only Google Ads report query via label routing."""
  tool_name = "ads_query_report"
  label = _resolve_label(label)
  caller_id, auth_token = _resolve_auth(caller_id, auth_token)
  try:
    config = _validated_context(
        label=label,
        caller_id=caller_id,
        auth_token=auth_token,
        tool_name=tool_name,
        expected_provider="google_ads",
    )
    token = _refresh_access_token(config)
    target_customer_id = customer_id or config.account_context.get("customer_id")
    if not target_customer_id:
      raise ToolError("customer_id is required for ads queries.")

    target_login_id = (
        login_customer_id
        or config.credentials.login_customer_id
        if config.credentials
        else None
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "developer-token": config.credentials.developer_token
        if config.credentials
        else "",
        "Content-Type": "application/json",
    }
    if target_login_id:
      headers["login-customer-id"] = target_login_id
    resp = httpx.post(
        (
            "https://googleads.googleapis.com/v19/customers/"
            f"{target_customer_id}/googleAds:searchStream"
        ),
        headers=headers,
        json={"query": query},
        timeout=45.0,
    )
    if resp.status_code != 200:
      raise ToolError("Google Ads query failed.")
    rows: list[dict[str, Any]] = []
    for batch in resp.json():
      rows.extend(batch.get("results", []))
    _audit_log(caller_id, label, tool_name, "success", f"rows={len(rows)}")
    return {"data": rows}
  except Exception as exc:  # pylint: disable=broad-exception-caught
    _audit_log(caller_id, label, tool_name, "error", str(exc))
    if isinstance(exc, ToolError):
      raise
    raise ToolError("ads_query_report failed.") from exc


@mcp.tool()
def analytics_run_report(
    metrics: list[str],
    dimensions: list[str],
    start_date: str,
    end_date: str,
    label: str | None = None,
    caller_id: str | None = None,
    auth_token: str | None = None,
    property_id: str | None = None,
) -> dict[str, Any]:
  """Runs a read-only GA4 report via label routing."""
  tool_name = "analytics_run_report"
  label = _resolve_label(label)
  caller_id, auth_token = _resolve_auth(caller_id, auth_token)
  try:
    config = _validated_context(
        label=label,
        caller_id=caller_id,
        auth_token=auth_token,
        tool_name=tool_name,
        expected_provider="ga4",
    )
    token = _refresh_access_token(config)
    target_property_id = property_id or config.account_context.get("property_id")
    if not target_property_id:
      raise ToolError("property_id is required for analytics reports.")
    payload = {
        "metrics": [{"name": x} for x in metrics],
        "dimensions": [{"name": x} for x in dimensions],
        "dateRanges": [{"startDate": start_date, "endDate": end_date}],
    }
    resp = httpx.post(
        (
            "https://analyticsdata.googleapis.com/v1beta/properties/"
            f"{target_property_id}:runReport"
        ),
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=45.0,
    )
    if resp.status_code != 200:
      raise ToolError("GA4 report query failed.")
    body = resp.json()
    _audit_log(
        caller_id, label, tool_name, "success", f"rows={len(body.get('rows', []))}"
    )
    return body
  except Exception as exc:  # pylint: disable=broad-exception-caught
    _audit_log(caller_id, label, tool_name, "error", str(exc))
    if isinstance(exc, ToolError):
      raise
    raise ToolError("analytics_run_report failed.") from exc


@mcp.tool()
def youtube_get_channel_stats(
    label: str | None = None,
    caller_id: str | None = None,
    auth_token: str | None = None,
    channel_id: str | None = None,
) -> dict[str, Any]:
  """Gets read-only YouTube channel stats via label routing."""
  tool_name = "youtube_get_channel_stats"
  label = _resolve_label(label)
  caller_id, auth_token = _resolve_auth(caller_id, auth_token)
  try:
    config = _validated_context(
        label=label,
        caller_id=caller_id,
        auth_token=auth_token,
        tool_name=tool_name,
        expected_provider="youtube",
    )
    token = _refresh_access_token(config)
    target_channel_id = channel_id or config.account_context.get("channel_id")
    params = {"part": "snippet,statistics"}
    if target_channel_id:
      params["id"] = target_channel_id
    else:
      params["mine"] = "true"
    resp = httpx.get(
        "https://www.googleapis.com/youtube/v3/channels",
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=30.0,
    )
    if resp.status_code != 200:
      raise ToolError("YouTube channel stats request failed.")
    body = resp.json()
    _audit_log(
        caller_id, label, tool_name, "success", f"items={len(body.get('items', []))}"
    )
    return body
  except Exception as exc:  # pylint: disable=broad-exception-caught
    _audit_log(caller_id, label, tool_name, "error", str(exc))
    if isinstance(exc, ToolError):
      raise
    raise ToolError("youtube_get_channel_stats failed.") from exc
