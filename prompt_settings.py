"""Front-End Editable System Prompt, Persona Presets, & KV-Cache Prefix Builder.

Stores user-editable prompt, persona, and inference settings in JSON inside
~/Library/Application Support/ARIES Public Preview Beta/aries_settings.json.
"""

import json
import os
from copy import deepcopy
from config import ARIES_DATA_DIR

SETTINGS_PATH = os.path.join(ARIES_DATA_DIR, "aries_settings.json")

PERSONA_PRESETS: dict[str, dict] = {
    "voice_assistant": {
        "id": "voice_assistant",
        "name": "Voice Assistant",
        "description": "Ultra-fast, natural spoken responses with zero markdown clutter.",
        "system_prompt": (
            "You are {assistant_name}, an intelligent, concise AI assistant running locally on macOS. "
            "Use clear markdown, fenced code blocks, LaTeX ($...$ / $$...$$), and Mermaid diagrams when helpful. "
            "If asked for a Mermaid diagram of your own architecture, output:\n"
            "```mermaid\nflowchart LR\n"
            "  UI[Apple HIG Web UI] --> ASR[Parakeet 0.6B ASR]\n"
            "  UI --> LLM[Gemma 4 E2B Metal]\n"
            "  ASR --> LLM\n"
            "  RAG[EmbeddingGemma 2 RAG] --> LLM\n"
            "  LLM --> TTS[Phonon 31M TTS]\n"
            "  LLM --> Canvas[Live Preview Canvas]\n```"
        ),
        "voice_system_prompt": (
            "You are {assistant_name}, an ultra-fast, natural voice assistant running locally on macOS. "
            "Speak clearly in 1 to 2 concise sentences without markdown, bullet points, or asterisks. "
            "Begin with a brief 2 to 4 word opening sentence ending in a period."
        ),
    },
    "software_engineer": {
        "id": "software_engineer",
        "name": "Software Engineer",
        "description": "Direct, production-grade code, architecture diagrams, and debugging.",
        "system_prompt": (
            "You are {assistant_name}, a pragmatic senior software engineer running locally on macOS. "
            "Provide clean, minimal, copy-ready code in fenced blocks, explain trade-offs concisely, "
            "and use Mermaid diagrams for architecture when helpful."
        ),
        "voice_system_prompt": (
            "You are {assistant_name}, a senior software engineer. "
            "Explain solutions clearly and concisely in 2 spoken sentences without markdown."
        ),
    },
    "research_analyst": {
        "id": "research_analyst",
        "name": "Research Analyst",
        "description": "Grounded synthesis, structured comparisons, and cited web/document answers.",
        "system_prompt": (
            "You are {assistant_name}, a thorough research analyst running locally on macOS. "
            "Ground answers strictly in provided documents and web sources, cite URLs when available, "
            "and use tables or LaTeX formulas when appropriate."
        ),
        "voice_system_prompt": (
            "You are {assistant_name}, a research analyst. "
            "Summarize key findings concisely in 2 spoken sentences without markdown."
        ),
    },
    "concise_editor": {
        "id": "concise_editor",
        "name": "Concise Editor",
        "description": "Tight, crisp prose and minimal token overhead.",
        "system_prompt": (
            "You are {assistant_name}, a crisp technical editor. "
            "Answer in the fewest clear words possible while preserving accuracy."
        ),
        "voice_system_prompt": (
            "You are {assistant_name}, a concise assistant. "
            "Answer in 1 short spoken sentence without markdown."
        ),
    },
    "tutor": {
        "id": "tutor",
        "name": "STEM Tutor",
        "description": "Step-by-step explanations with KaTeX math and visual Mermaid diagrams.",
        "system_prompt": (
            "You are {assistant_name}, a patient STEM tutor running locally on macOS. "
            "Break concepts down step by step, format equations with LaTeX ($...$ / $$...$$), "
            "and use Mermaid diagrams to illustrate processes."
        ),
        "voice_system_prompt": (
            "You are {assistant_name}, a helpful tutor. "
            "Explain the core idea clearly in 2 spoken sentences without markdown."
        ),
    },
}

DEFAULT_SETTINGS: dict = {
    "assistant_name": "ARIES",
    "user_display_name": "User",
    "persona_preset": "voice_assistant",
    "prompt_mode": "chat",  # "chat" | "voice" | "custom"
    "system_prompt": PERSONA_PRESETS["voice_assistant"]["system_prompt"],
    "voice_system_prompt": PERSONA_PRESETS["voice_assistant"]["voice_system_prompt"],
    "temperature": 0.7,
    "max_history_turns": 6,
    "enable_thinking": False,
    "voice_enabled": True,
    "local_api_bridge_enabled": False,
    "theme": "dark",
}


class PromptSettingsManager:
    def __init__(self, path: str = SETTINGS_PATH):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._settings = self._load()

    def _load(self) -> dict:
        data = deepcopy(DEFAULT_SETTINGS)
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    for k, v in loaded.items():
                        if k in data:
                            data[k] = v
            except Exception:
                pass
        return data

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self._settings, fh, indent=2, ensure_ascii=False)

    def get(self) -> dict:
        return deepcopy(self._settings)

    def update(self, patch: dict) -> dict:
        for k, v in patch.items():
            if k not in DEFAULT_SETTINGS:
                continue
            if k == "temperature":
                self._settings[k] = max(0.0, min(2.0, float(v)))
            elif k == "max_history_turns":
                self._settings[k] = max(1, min(30, int(v)))
            elif k in ("enable_thinking", "voice_enabled", "local_api_bridge_enabled"):
                self._settings[k] = bool(v)
            else:
                self._settings[k] = str(v)
        self._save()
        return self.get()

    def reset(self) -> dict:
        self._settings = deepcopy(DEFAULT_SETTINGS)
        self._save()
        return self.get()

    def build_static_system_prompt(
        self,
        profile_text: str = "",
        instincts_text: str = "",
        is_voice_turn: bool = False,
    ) -> str:
        """Builds a strictly static, KV-cache-friendly system prefix (no dynamic timestamps)."""
        s = self._settings
        assistant = (s.get("assistant_name") or "ARIES").strip()
        user_name = (s.get("user_display_name") or "User").strip()

        raw_tpl = s["voice_system_prompt"] if (is_voice_turn or s.get("prompt_mode") == "voice") else s["system_prompt"]
        base = raw_tpl.replace("{assistant_name}", assistant).replace("{user_name}", user_name)

        parts = [base]
        if user_name and user_name.lower() != "user":
            parts.append(f"The user's preferred name is {user_name}.")
        if profile_text.strip():
            parts.append(profile_text.strip())
        if instincts_text.strip():
            parts.append(instincts_text.strip())
        return "\n\n".join(parts)


prompt_settings = PromptSettingsManager()
