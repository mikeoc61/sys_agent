#!/usr/bin/env python3
"""Compare sys_agent's model catalogue with provider inventories.

Prints one JSON report for a monitoring agent. Only GET requests are sent to
providers; the sole local write is an atomic, private inventory snapshot.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import sys_agent as agent  # noqa: E402

SOURCES = {
    "openai": "https://platform.openai.com/docs/api-reference/models/list",
    "anthropic": "https://platform.claude.com/docs/en/api/models/list",
    "deepseek": "https://api-docs.deepseek.com/api/list-models/",
}
KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def state_path() -> Path:
    base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    return base / "sys_agent/model_watch.json"


def fetch_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        data = json.load(response)
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise ValueError("unexpected model-list response")
    return data


def inventory(provider: str, key: str) -> dict[str, dict[str, Any]]:
    if provider == "openai":
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {key}"}
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/models?limit=1000"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
        if workspace := os.environ.get("ANTHROPIC_WORKSPACE_ID"):
            headers["anthropic-workspace-id"] = workspace
    else:
        url = "https://api.deepseek.com/models"
        headers = {"Authorization": f"Bearer {key}"}

    result: dict[str, dict[str, Any]] = {}
    while True:
        page = fetch_json(url, headers)
        for entry in page["data"]:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                raise ValueError("unexpected model entry")
            model_id = entry["id"]
            if model_id in result:
                raise ValueError("duplicate model id in provider inventory")
            # These fields identify revision/capability changes without storing
            # provider responses or any account data in the state file.
            result[model_id] = {
                field: entry[field]
                for field in ("name", "display_name", "created", "created_at",
                              "context_window", "max_input_tokens",
                              "max_output_tokens", "max_tokens", "capabilities",
                              "api_capabilities")
                if field in entry
            }
        if provider != "anthropic" or not page.get("has_more"):
            break
        cursor = page.get("last_id")
        if not isinstance(cursor, str) or not cursor:
            raise ValueError("Anthropic pagination has no last_id")
        url = "https://api.anthropic.com/v1/models?" + urllib.parse.urlencode(
            {"limit": 1000, "after_id": cursor})
    return result


def load_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError(f"invalid snapshot: {path}")
    return data


def save_state(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".model_watch-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def compare(provider: str, current: dict[str, dict[str, Any]],
            previous: dict[str, dict[str, Any]] | None,
            catalogue_changed: bool = False) -> dict[str, Any]:
    listed = set(agent.PROVIDER_MODELS[provider])
    visible = set(current)
    defaults = agent.DEFAULT_MODELS[provider]
    result: dict[str, Any] = {
        "source": SOURCES[provider],
        "visible_count": len(visible),
        "curated_models": sorted(listed),
        "default_model": defaults,
        "listed_not_visible": sorted(listed - visible),
        "default_not_visible": defaults not in visible,
        "context_review": [],
        "new_models": [],
        "disappeared_models": [],
        "metadata_changes": [],
    }
    for model_id in sorted(listed & visible):
        upstream = current[model_id].get("context_window")
        if upstream is None:
            upstream = current[model_id].get("max_input_tokens")
        local = agent.CONTEXT_WINDOWS.get(model_id)
        # sys_agent rounds some 1,048,576-token windows to 1,000,000.
        if isinstance(upstream, int) and local and abs(upstream - local) / upstream > 0.10:
            result["context_review"].append(
                {"model": model_id, "local": local, "provider": upstream})
    if previous is not None:
        result["new_models"] = [
            {"id": name, "metadata": current[name],
             "name_prefix_match": name.startswith(agent.PROVIDER_MODEL_PREFIXES[provider])}
            for name in sorted(visible - set(previous))
        ]
        result["disappeared_models"] = sorted(set(previous) - visible)
        result["metadata_changes"] = [
            {"id": name, "before": previous[name], "after": current[name]}
            for name in sorted(visible & set(previous))
            if previous[name] != current[name]
        ]
    result["needs_review"] = bool(
        result["listed_not_visible"] or result["default_not_visible"]
        or result["context_review"] or result["new_models"]
        or result["disappeared_models"] or result["metadata_changes"])
    result["new_signal"] = bool(
        result["new_models"] or result["disappeared_models"]
        or result["metadata_changes"]
        or ((previous is None or catalogue_changed)
            and (result["listed_not_visible"] or result["default_not_visible"]
                 or result["context_review"])))
    return result


def run(path: Path, *, write_state: bool = True) -> dict[str, Any]:
    env_file = agent.find_env_file(os.environ.get("SYS_ENV_FILE"))
    if env_file:
        agent.load_env_file(env_file)
        agent.init_config()
    old = load_state(path)
    old_inventories = old.get("inventories", {})
    if not isinstance(old_inventories, dict):
        raise ValueError(f"invalid inventory snapshot: {path}")
    catalogue = {
        provider: {"default": agent.DEFAULT_MODELS[provider],
                   "listed": list(agent.PROVIDER_MODELS[provider]),
                   "context": {name: agent.CONTEXT_WINDOWS.get(name)
                               for name in agent.PROVIDER_MODELS[provider]}}
        for provider in KEYS
    }
    old_catalogue = old.get("catalogue", {})
    if not isinstance(old_catalogue, dict):
        raise ValueError(f"invalid catalogue snapshot: {path}")
    old_health = old.get("health", {})
    if not isinstance(old_health, dict):
        raise ValueError(f"invalid provider health snapshot: {path}")
    saved_catalogue = dict(old_catalogue)
    saved_health = dict(old_health)
    now = datetime.now(timezone.utc).isoformat()
    new_inventories = dict(old_inventories)
    results: dict[str, Any] = {}
    attempted = False
    for provider, env_name in KEYS.items():
        key = os.environ.get(env_name)
        if not key:
            results[provider] = {"status": "unchecked", "reason": f"{env_name} unavailable"}
            continue
        attempted = True
        previous_health = old_health.get(provider, {})
        if not isinstance(previous_health, dict):
            raise ValueError("invalid provider health snapshot")
        try:
            current = inventory(provider, key)
            previous = old_inventories.get(provider)
            if previous is not None and not isinstance(previous, dict):
                raise ValueError("invalid provider snapshot")
            results[provider] = {
                "status": "baseline" if previous is None else "checked",
                "consecutive_failures": 0,
                "last_success_at": now,
                "last_error_at": previous_health.get("last_error_at"),
                **compare(provider, current, previous,
                          catalogue_changed=(old_catalogue.get(provider)
                                             != catalogue[provider])),
            }
            new_inventories[provider] = current
            saved_catalogue[provider] = catalogue[provider]
            saved_health[provider] = {
                "consecutive_failures": 0,
                "last_success_at": now,
                "last_error_at": previous_health.get("last_error_at"),
            }
        except urllib.error.HTTPError as exc:
            failures = previous_health.get("consecutive_failures", 0) + 1
            results[provider] = {
                "status": "error", "reason": f"HTTP {exc.code}",
                "consecutive_failures": failures,
                "persistent_failure": failures >= 2,
                "last_success_at": previous_health.get("last_success_at"),
                "last_error_at": now,
            }
            saved_health[provider] = {
                "consecutive_failures": failures,
                "last_success_at": previous_health.get("last_success_at"),
                "last_error_at": now,
            }
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            # Never print response bodies, request headers, or credentials.
            failures = previous_health.get("consecutive_failures", 0) + 1
            results[provider] = {
                "status": "error", "reason": type(exc).__name__,
                "consecutive_failures": failures,
                "persistent_failure": failures >= 2,
                "last_success_at": previous_health.get("last_success_at"),
                "last_error_at": now,
            }
            saved_health[provider] = {
                "consecutive_failures": failures,
                "last_success_at": previous_health.get("last_success_at"),
                "last_error_at": now,
            }
    if write_state and attempted:
        save_state(path, {"version": 1, "checked_at": now,
                          "inventories": new_inventories,
                          "catalogue": saved_catalogue,
                          "health": saved_health})
    return {
        "schema_version": 1,
        "checked_at": now,
        "snapshot_path": str(path),
        "providers": results,
        "needs_review": any(item.get("needs_review", False)
                            for item in results.values()),
        "new_signal": any(item.get("new_signal", False)
                          for item in results.values()),
        "coverage_incomplete": any(item["status"] in ("unchecked", "error")
                                   for item in results.values()),
        "coverage_needs_repair": any(item.get("persistent_failure", False)
                                     for item in results.values()),
        "checked_providers": sum(item["status"] in ("baseline", "checked")
                                 for item in results.values()),
        "interpretation": (
            "New IDs and metadata changes are leads, not proof of sys_agent "
            "compatibility. A missing ID may reflect account access. Check "
            "official release notes, pricing and tool-call behavior before "
            "recommending a model-list or default change."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=state_path(),
                        help="private snapshot path")
    parser.add_argument("--no-state", action="store_true",
                        help="compare without updating the snapshot")
    args = parser.parse_args()
    try:
        report = run(args.state, write_state=not args.no_state)
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema_version": 1, "status": "error",
                          "reason": str(exc)}))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
