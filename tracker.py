"""Usage and Rate Limit Tracker for Gemma Live Voice Agent.

Tracks daily request counts, token consumption, and session stats with automatic
rollover at Midnight Pacific Time (PT). Supports both local LM Studio (unlimited)
and Cloud Gemma (1,500 requests/day).
"""

import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

PACIFIC_TZ = timezone(timedelta(hours=-8))
DEFAULT_TRACKER_FILE = os.path.join(os.path.dirname(__file__), "usage_tracker.json")

MODEL_LIMITS = {
    "local-gemma-4-e2b": 0,          # 0 = Unlimited local
    "google/gemma-4-e2b-it-qat": 0,  # 0 = Unlimited local
    "gemma-4-26b-a4b-it": 1500,
    "gemma-4-31b-it": 1500,
}

class UsageTracker:
    """Manages persistent tracking of API turns and tokens."""

    def __init__(self, filepath: str = DEFAULT_TRACKER_FILE):
        self.filepath = filepath
        self.data = self._load()

    def _get_current_pt_date(self) -> str:
        """Returns the current date in Pacific Time (YYYY-MM-DD)."""
        return datetime.now(PACIFIC_TZ).strftime("%Y-%m-%d")

    def _get_model_limit(self, model: str) -> int:
        if model.startswith("local-") or "qat" in model:
            return 0
        return MODEL_LIMITS.get(model, 1500)

    def _load(self) -> Dict[str, Any]:
        """Loads data from disk, resetting if a new quota day has started."""
        current_date = self._get_current_pt_date()
        
        default_data = {
            "date": current_date,
            "requests_today": 0,
            "sessions_today": 0,
            "prompt_tokens_today": 0,
            "response_tokens_today": 0,
            "total_tokens_today": 0,
            "models": {},
            "history": {}
        }

        if not os.path.exists(self.filepath):
            return default_data

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            data.setdefault("models", {})
            data.setdefault("history", {})

            stored_date = data.get("date")
            if stored_date != current_date:
                data["history"][stored_date] = {
                    "requests_total": data.get("requests_today", 0),
                    "tokens_total": data.get("total_tokens_today", 0),
                    "models": data.get("models", {}),
                }
                data["date"] = current_date
                data["requests_today"] = 0
                data["sessions_today"] = 0
                data["prompt_tokens_today"] = 0
                data["response_tokens_today"] = 0
                data["total_tokens_today"] = 0
                data["models"] = {}
                self._save_raw(data)

            return data
        except Exception:
            return default_data

    def _ensure_model(self, model: str) -> Dict[str, Any]:
        """Ensures the per-model entry exists in today's tracker data."""
        if "models" not in self.data:
            self.data["models"] = {}
        if model not in self.data["models"]:
            self.data["models"][model] = {
                "requests_today": 0,
                "daily_limit": self._get_model_limit(model),
                "prompt_tokens_today": 0,
                "response_tokens_today": 0,
                "total_tokens_today": 0,
            }
        return self.data["models"][model]

    def _save_raw(self, data: Dict[str, Any]):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _save(self):
        self._save_raw(self.data)

    def record_session(self, model: str = "local-gemma-4-e2b"):
        self.data["sessions_today"] = self.data.get("sessions_today", 0) + 1
        self._ensure_model(model)
        self._save()

    def record_turn(
        self,
        prompt_tokens: int = 0,
        response_tokens: int = 0,
        count_as_request: bool = True,
        model: str = "local-gemma-4-e2b",
    ) -> Dict[str, Any]:
        current_date = self._get_current_pt_date()
        if self.data.get("date") != current_date:
            self.data = self._load()

        model_entry = self._ensure_model(model)

        if count_as_request:
            self.data["requests_today"] = self.data.get("requests_today", 0) + 1
            model_entry["requests_today"] = model_entry.get("requests_today", 0) + 1

        self.data["prompt_tokens_today"] = self.data.get("prompt_tokens_today", 0) + prompt_tokens
        self.data["response_tokens_today"] = self.data.get("response_tokens_today", 0) + response_tokens
        self.data["total_tokens_today"] = self.data.get("prompt_tokens_today", 0) + self.data.get("response_tokens_today", 0)

        model_entry["prompt_tokens_today"] = model_entry.get("prompt_tokens_today", 0) + prompt_tokens
        model_entry["response_tokens_today"] = model_entry.get("response_tokens_today", 0) + response_tokens
        model_entry["total_tokens_today"] = model_entry.get("prompt_tokens_today", 0) + model_entry.get("response_tokens_today", 0)

        self._save()
        return self.get_status(model=model)

    def get_status(self, model: str = "local-gemma-4-e2b") -> Dict[str, Any]:
        model_entry = self._ensure_model(model)
        used = model_entry.get("requests_today", 0)
        limit = model_entry.get("daily_limit", self._get_model_limit(model))
        
        if limit == 0:
            remaining = 999999
            percent = 0.0
            is_exhausted = False
        else:
            remaining = max(0, limit - used)
            percent = min(100.0, (used / limit * 100.0))
            is_exhausted = used >= limit

        return {
            "date": self.data.get("date"),
            "model": model,
            "used": used,
            "limit": limit,
            "remaining": remaining,
            "percent": percent,
            "prompt_tokens": model_entry.get("prompt_tokens_today", 0),
            "response_tokens": model_entry.get("response_tokens_today", 0),
            "total_tokens": model_entry.get("total_tokens_today", 0),
            "is_exhausted": is_exhausted,
            "global_total_requests": self.data.get("requests_today", 0),
        }
