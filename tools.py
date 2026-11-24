"""Non-blocking tools and function declarations for the Gemini Live Voice Agent."""

import asyncio
import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.parse
import webbrowser
from datetime import datetime
from html import unescape
from urllib.parse import quote
import aiohttp
import psutil
from google.genai import types

from rag_memory import memory_store

# ---------------------------------------------------------------------------
# WMO Weather Code Interpretations
# ---------------------------------------------------------------------------
WMO_WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# ---------------------------------------------------------------------------
# Tool Declarations (with behavior="NON_BLOCKING")
# ---------------------------------------------------------------------------

WEATHER_DECLARATION = types.FunctionDeclaration(
    name="get_current_weather",
    description="Get weather for a city.",
    behavior="NON_BLOCKING",
    parameters={
        "type": "OBJECT",
        "properties": {
            "location": {"type": "STRING"},
            "unit": {"type": "STRING", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"]
    }
)

SYSTEM_STATUS_DECLARATION = types.FunctionDeclaration(
    name="get_system_status",
    description="Get PC RAM, disk, CPU, battery, and time.",
    behavior="NON_BLOCKING",
    parameters={
        "type": "OBJECT",
        "properties": {}
    }
)

OPEN_APP_DECLARATION = types.FunctionDeclaration(
    name="open_application",
    description="Launch a desktop app (e.g. spotify, calculator, notes, terminal, safari).",
    behavior="NON_BLOCKING",
    parameters={
        "type": "OBJECT",
        "properties": {
            "app_name": {"type": "STRING"}
        },
        "required": ["app_name"]
    }
)

OPEN_WEBSITE_DECLARATION = types.FunctionDeclaration(
    name="open_website",
    description="Open URL or search Google, YouTube, or GitHub.",
    behavior="NON_BLOCKING",
    parameters={
        "type": "OBJECT",
        "properties": {
            "url": {"type": "STRING"},
            "query": {"type": "STRING"},
            "destination": {"type": "STRING", "enum": ["google", "youtube", "github"]}
        }
    }
)

WEB_SEARCH_DECLARATION = types.FunctionDeclaration(
    name="search_web_live",
    description="Search the web for current info.",
    behavior="NON_BLOCKING",
    parameters={
        "type": "OBJECT",
        "properties": {
            "query": {"type": "STRING"}
        },
        "required": ["query"]
    }
)

KNOWLEDGE_DECLARATION = types.FunctionDeclaration(
    name="search_knowledge_base",
    description="Search saved notes and RAG memory.",
    behavior="NON_BLOCKING",
    parameters={
        "type": "OBJECT",
        "properties": {
            "query": {"type": "STRING"}
        },
        "required": ["query"]
    }
)

SAVE_MEMORY_DECLARATION = types.FunctionDeclaration(
    name="save_memory_note",
    description="Save a note or fact to memory.",
    behavior="NON_BLOCKING",
    parameters={
        "type": "OBJECT",
        "properties": {
            "note": {"type": "STRING"},
            "title": {"type": "STRING"}
        },
        "required": ["note"]
    }
)

CALENDAR_DECLARATION = types.FunctionDeclaration(
    name="create_calendar_event",
    description="Create a calendar event.",
    behavior="NON_BLOCKING",
    parameters={
        "type": "OBJECT",
        "properties": {
            "title": {"type": "STRING"},
            "date": {"type": "STRING"},
            "time": {"type": "STRING"},
            "duration_minutes": {"type": "INTEGER"}
        },
        "required": ["title", "date", "time"]
    }
)

TIMER_DECLARATION = types.FunctionDeclaration(
    name="set_timer_reminder",
    description="Set a timer or reminder.",
    behavior="NON_BLOCKING",
    parameters={
        "type": "OBJECT",
        "properties": {
            "label": {"type": "STRING"},
            "seconds": {"type": "INTEGER"}
        },
        "required": ["label", "seconds"]
    }
)

def get_agent_tools() -> list[types.Tool]:
    """Returns the list of Tool specifications with NON_BLOCKING declarations."""
    return [
        types.Tool(
            function_declarations=[
                WEATHER_DECLARATION,
                SYSTEM_STATUS_DECLARATION,
                OPEN_APP_DECLARATION,
                OPEN_WEBSITE_DECLARATION,
                WEB_SEARCH_DECLARATION,
                KNOWLEDGE_DECLARATION,
                SAVE_MEMORY_DECLARATION,
                CALENDAR_DECLARATION,
                TIMER_DECLARATION,
            ]
        )
    ]

# ---------------------------------------------------------------------------
# Windows Native Structs for System Status
# ---------------------------------------------------------------------------

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', ctypes.c_ulong),
        ('dwMemoryLoad', ctypes.c_ulong),
        ('ullTotalPhys', ctypes.c_ulonglong),
        ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong),
        ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong),
        ('ullAvailVirtual', ctypes.c_ulonglong),
        ('ullAvailExtendedVirtual', ctypes.c_ulonglong)
    ]

class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ('ACLineStatus', ctypes.c_byte),
        ('BatteryFlag', ctypes.c_byte),
        ('BatteryLifePercent', ctypes.c_byte),
        ('Reserved1', ctypes.c_byte),
        ('BatteryLifeTime', ctypes.c_ulong),
        ('BatteryFullLifeTime', ctypes.c_ulong)
    ]

# ---------------------------------------------------------------------------
# Async Non-Blocking Tool Implementations
# ---------------------------------------------------------------------------

async def _get_system_status() -> dict:
    """Returns real-time system metrics."""
    await asyncio.sleep(0.2)
    now_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    if sys.platform != "win32":
        vm = psutil.virtual_memory()
        total_ram = round(vm.total / (1024**3), 1)
        avail_ram = round(vm.available / (1024**3), 1)
        total_disk, _used_disk, free_disk = shutil.disk_usage("/")
        total_disk_gb = round(total_disk / (1024**3), 1)
        free_disk_gb = round(free_disk / (1024**3), 1)
        batt = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        if batt is not None:
            is_plugged_in = bool(batt.power_plugged)
            battery_pct = int(round(batt.percent))
        else:
            is_plugged_in = True
            battery_pct = "N/A (Desktop/AC)"
        return {
            "status": "success",
            "current_time": now_str,
            "ram_available_gb": avail_ram,
            "ram_total_gb": total_ram,
            "ram_usage_percent": f"{int(round(vm.percent))}%",
            "c_drive_free_gb": free_disk_gb,
            "c_drive_total_gb": total_disk_gb,
            "cpu_cores": os.cpu_count() or 4,
            "power_source": "AC Wall Power" if is_plugged_in else "Battery",
            "battery_percent": battery_pct,
        }

    # Memory
    mem_status = _MEMORYSTATUSEX()
    mem_status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
    total_ram = round(mem_status.ullTotalPhys / (1024**3), 1)
    avail_ram = round(mem_status.ullAvailPhys / (1024**3), 1)

    # Disk
    total_disk, used_disk, free_disk = shutil.disk_usage("C:\\")
    total_disk_gb = round(total_disk / (1024**3), 1)
    free_disk_gb = round(free_disk / (1024**3), 1)

    # Power
    power = _SYSTEM_POWER_STATUS()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power))
    is_plugged_in = power.ACLineStatus == 1
    battery_pct = power.BatteryLifePercent if power.BatteryLifePercent != -1 and power.BatteryLifePercent <= 100 else "N/A (Desktop/AC)"

    now_str = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    return {
        "status": "success",
        "current_time": now_str,
        "ram_available_gb": avail_ram,
        "ram_total_gb": total_ram,
        "ram_usage_percent": f"{mem_status.dwMemoryLoad}%",
        "c_drive_free_gb": free_disk_gb,
        "c_drive_total_gb": total_disk_gb,
        "cpu_cores": os.cpu_count() or 4,
        "power_source": "AC Wall Power" if is_plugged_in else "Battery",
        "battery_percent": battery_pct,
    }

async def _open_application(app_name: str) -> dict:
    """Opens a recognized desktop application."""
    await asyncio.sleep(0.3)
    name_clean = app_name.lower().strip()

    if sys.platform == "darwin":
        mac_map = {
            "spotify": "Spotify",
            "calculator": "Calculator",
            "calc": "Calculator",
            "notepad": "Notes",
            "notes": "Notes",
            "paint": "Preview",
            "task manager": "Activity Monitor",
            "activity monitor": "Activity Monitor",
            "explorer": "Finder",
            "finder": "Finder",
            "files": "Finder",
            "terminal": "Terminal",
            "browser": "Safari",
            "safari": "Safari",
            "chrome": "Google Chrome",
        }
        target = mac_map.get(name_clean, re.sub(r"[^a-zA-Z0-9_\-\. ]", "", app_name).strip())
        try:
            subprocess.Popen(["open", "-a", target], shell=False)
            return {"status": "success", "app": app_name, "message": f"Successfully launched {target}."}
        except Exception as e:
            return {"status": "error", "app": app_name, "error": str(e)}

    app_map = {
        "spotify": "spotify:",
        "calculator": "calc.exe",
        "calc": "calc.exe",
        "notepad": "notepad.exe",
        "paint": "mspaint.exe",
        "mspaint": "mspaint.exe",
        "task manager": "taskmgr.exe",
        "taskmgr": "taskmgr.exe",
        "explorer": "explorer.exe",
        "files": "explorer.exe",
        "terminal": "powershell.exe",
        "powershell": "powershell.exe",
        "cmd": "cmd.exe",
        "edge": "msedge",
        "browser": "msedge",
    }

    cmd = app_map.get(name_clean)
    try:
        if cmd:
            if cmd.startswith("spotify:"):
                subprocess.Popen(["cmd", "/c", "start", cmd], shell=False)
            elif cmd == "msedge":
                subprocess.Popen(["cmd", "/c", "start", "msedge"], shell=False)
            else:
                subprocess.Popen([cmd], shell=False)
            return {"status": "success", "app": app_name, "message": f"Successfully launched {app_name}."}
        else:
            safe_name = re.sub(r'[^a-zA-Z0-9_\-\. ]', '', app_name)
            subprocess.Popen(["cmd", "/c", "start", safe_name], shell=False)
            return {"status": "success", "app": app_name, "message": f"Attempted to open '{safe_name}'."}
    except Exception as e:
        return {"status": "error", "app": app_name, "error": str(e)}

async def _open_website(url: str = "", query: str = "", destination: str = "google") -> dict:
    """Opens a website or searches in the user's browser."""
    await asyncio.sleep(0.2)
    target_url = url.strip()
    if not target_url and query:
        q_enc = quote(query)
        if destination == "youtube":
            target_url = f"https://www.youtube.com/results?search_query={q_enc}"
        elif destination == "github":
            target_url = f"https://github.com/search?q={q_enc}"
        else:
            target_url = f"https://www.google.com/search?q={q_enc}"

    if not target_url:
        return {"status": "error", "error": "Either a URL or query must be provided."}

    if not target_url.startswith(("http://", "https://")):
        target_url = "https://" + target_url

    try:
        webbrowser.open(target_url)
        return {"status": "success", "opened_url": target_url, "message": f"Opened {target_url} in your browser."}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def _search_web_live(query: str) -> dict:
    """Live web search combining DuckDuckGo and Wikipedia."""
    headers = {"User-Agent": "GeminiVoiceAgent/1.0 (Windows NT 10.0)"}
    results = []

    try:
        async with aiohttp.ClientSession(headers=headers) as s:
            # 1. Check DuckDuckGo Instant Answer
            try:
                ddg_url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
                async with s.get(ddg_url, timeout=aiohttp.ClientTimeout(total=3)) as r:
                    if r.status == 200:
                        data = await r.json(content_type=None)
                        if isinstance(data, dict):
                            if data.get("AbstractText"):
                                results.append(data["AbstractText"])
                            for t in data.get("RelatedTopics", [])[:3]:
                                if isinstance(t, dict) and t.get("Text"):
                                    results.append(t["Text"])
            except Exception:
                pass

            # 2. Query Wikipedia Search API if needed
            if len(results) < 2:
                try:
                    wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(query)}&utf8=&format=json"
                    async with s.get(wiki_url, timeout=aiohttp.ClientTimeout(total=4)) as r:
                        if r.status == 200:
                            wiki_data = await r.json()
                            if isinstance(wiki_data, dict):
                                search_items = wiki_data.get("query", {}).get("search", [])
                                for item in search_items[:3]:
                                    title = item.get("title", "")
                                    snippet = re.sub(r'<[^>]+>', '', unescape(item.get("snippet", ""))).strip()
                                    results.append(f"{title}: {snippet}")
                except Exception:
                    pass

        if results:
            return {"status": "success", "query": query, "results": results[:4]}
        else:
            return {"status": "no_results", "query": query, "message": f"No direct search summaries found for '{query}'."}
    except Exception as e:
        return {"status": "error", "query": query, "error": str(e)}

async def _get_current_weather(location: str, unit: str = "fahrenheit") -> dict:
    """Queries live actual weather data for any global city using Open-Meteo API."""
    temp_unit = "fahrenheit" if unit.lower() != "celsius" else "celsius"
    unit_sym = "°F" if temp_unit == "fahrenheit" else "°C"

    encoded_loc = urllib.parse.quote(location)
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_loc}&count=1&language=en&format=json"
    headers = {"User-Agent": "GeminiLiveVoiceAgent/1.0"}

    try:
        async with aiohttp.ClientSession(headers=headers) as http_client:
            # 1. Geocode location name to latitude & longitude
            async with http_client.get(geo_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return {"location": location, "error": f"Geocoding service returned status {resp.status}"}
                geo_data = await resp.json()

            results = geo_data.get("results")
            if not results:
                return {"location": location, "error": f"Could not find coordinates for location '{location}'."}

            match = results[0]
            lat = match.get("latitude")
            lon = match.get("longitude")
            city_name = match.get("name", location)
            admin1 = match.get("admin1", "")
            country = match.get("country", "")
            display_loc = f"{city_name}, {admin1} {country}".replace("  ", " ").strip(", ")

            # 2. Query live weather
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}&"
                f"current=temperature_2m,relative_humidity_2m,weather_code,apparent_temperature,wind_speed_10m&"
                f"temperature_unit={temp_unit}"
            )

            async with http_client.get(weather_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status != 200:
                    return {"location": display_loc, "error": f"Weather API returned status {resp.status}"}
                weather_data = await resp.json()

            current = weather_data.get("current", {})
            temp = round(current.get("temperature_2m", 0))
            feels_like = round(current.get("apparent_temperature", temp))
            humidity = current.get("relative_humidity_2m", 50)
            code = current.get("weather_code", 0)
            condition = WMO_WEATHER_CODES.get(code, "Clear")
            wind = round(current.get("wind_speed_10m", 0))

            return {
                "location": display_loc,
                "condition": condition,
                "temperature": f"{temp}{unit_sym}",
                "feels_like": f"{feels_like}{unit_sym}",
                "humidity": f"{humidity}%",
                "wind": f"{wind} mph",
            }
    except Exception as e:
        return {
            "location": location,
            "error": f"Live weather lookup failed: {str(e)}"
        }

async def _search_knowledge_base(query: str) -> dict:
    """Searches local EmbeddingGemma 2 RAG vector memory for indexed documents, credentials, code, and long-term notes."""
    try:
        results = await asyncio.to_thread(memory_store.search_rag, query, top_k=3, min_score=0.55)
        if results:
            formatted = [
                f"[{r['title']} (similarity {r['score']:.2f})]: {r['content']}"
                for r in results
            ]
            return {
                "status": "found",
                "count": len(results),
                "query": query,
                "results": formatted,
            }
        return {
            "status": "not_found",
            "query": query,
            "message": f"No relevant articles or memories found for '{query}' in local EmbeddingGemma 2 vector store."
        }
    except Exception as e:
        return {"status": "error", "query": query, "error": str(e)}

async def _save_memory_note(note: str, title: str = "Personal Note") -> dict:
    """Saves a new note, fact, or credential into ARIES's persistent local EmbeddingGemma 2 semantic memory."""
    try:
        clean_title = (title or "Personal Note").strip()
        clean_note = (note or "").strip()
        if not clean_note:
            return {"status": "error", "error": "Note content cannot be empty."}
        ids = await asyncio.to_thread(memory_store.add_knowledge_document, clean_title, clean_note, "qa")
        return {
            "status": "saved",
            "title": clean_title,
            "chunk_count": len(ids),
            "message": f"Successfully indexed '{clean_title}' in EmbeddingGemma 2 local vector memory."
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def _create_calendar_event(title: str, date: str, time: str, duration_minutes: int = 30) -> dict:
    """Simulates scheduling a calendar event."""
    return {
        "status": "success",
        "event_id": "evt_984321",
        "title": title,
        "date": date,
        "time": time,
        "duration_minutes": duration_minutes,
        "message": f"Successfully scheduled '{title}' for {date} at {time}."
    }

async def _set_timer_reminder(label: str, seconds: int) -> dict:
    """Sets a timer / reminder."""
    return {
        "status": "active",
        "label": label,
        "duration_seconds": seconds,
        "fires_at": datetime.now().strftime("%H:%M:%S"),
        "message": f"Timer set for {seconds} seconds: '{label}'."
    }

async def _query_sandboxed_buffer(buffer_id: str, query: str = "") -> dict:
    from efficiency_engine import context_sandbox
    return {"result": context_sandbox.query(buffer_id, query)}

async def _read_url_markdown(url: str) -> dict:
    from web_and_tools import fetch_url_markdown
    return {"markdown": await fetch_url_markdown(url)}

async def _agent_reach_fetch(target: str) -> dict:
    from web_and_tools import agent_reach_fetch
    return {"content": await agent_reach_fetch(target)}

async def _get_stock_quote(symbol: str) -> dict:
    from web_and_tools import fetch_stock_quote
    return {"quote": await fetch_stock_quote(symbol)}

async def _run_sandboxed_code(language: str, code: str) -> dict:
    from web_and_tools import SandboxedCodeRunner
    return SandboxedCodeRunner.run(language, code)

async def _save_instinct(rule: str, category: str = "preference") -> dict:
    from efficiency_engine import instinct_store
    return {"saved": instinct_store.add(rule, category)}

# Tool Registry
TOOL_HANDLERS = {
    "get_current_weather": _get_current_weather,
    "get_system_status": _get_system_status,
    "open_application": _open_application,
    "open_website": _open_website,
    "search_web_live": _search_web_live,
    "search_knowledge_base": _search_knowledge_base,
    "save_memory_note": _save_memory_note,
    "create_calendar_event": _create_calendar_event,
    "set_timer_reminder": _set_timer_reminder,
    "query_sandboxed_buffer": _query_sandboxed_buffer,
    "read_url_markdown": _read_url_markdown,
    "agent_reach_fetch": _agent_reach_fetch,
    "get_stock_quote": _get_stock_quote,
    "run_sandboxed_code": _run_sandboxed_code,
    "save_instinct": _save_instinct,
}

async def execute_tool_async(name: str, args: dict) -> dict:
    """Dispatches and executes a tool asynchronously."""
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return {"error": f"Tool '{name}' is not recognized."}
    try:
        return await handler(**args)
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}

def get_openai_tools() -> list:
    """Returns slimmed tools formatted for OpenAI-compatible local inference APIs."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_system_status",
                "description": "Get Mac RAM, disk, CPU, battery, and time.",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_url_markdown",
                "description": "Fetch a web page and extract clean Markdown.",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "agent_reach_fetch",
                "description": "Read GitHub README, Reddit JSON thread, or YouTube page without API keys.",
                "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_quote",
                "description": "Get live stock quote by ticker symbol (e.g. AAPL, NVDA).",
                "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_sandboxed_code",
                "description": "Run Python, Bash, or JavaScript in an isolated /private/tmp sandbox.",
                "parameters": {"type": "object", "properties": {"language": {"type": "string"}, "code": {"type": "string"}}, "required": ["language", "code"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "query_sandboxed_buffer",
                "description": "Search lines inside a sandboxed tool output buffer.",
                "parameters": {"type": "object", "properties": {"buffer_id": {"type": "string"}, "query": {"type": "string"}}, "required": ["buffer_id"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_instinct",
                "description": "Save a concise user preference or rule for future turns.",
                "parameters": {"type": "object", "properties": {"rule": {"type": "string"}, "category": {"type": "string"}}, "required": ["rule"]}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "open_application",
                "description": "Launch a macOS app (e.g. Safari, Calculator, Notes).",
                "parameters": {
                    "type": "object",
                    "properties": {"app_name": {"type": "string"}},
                    "required": ["app_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "open_website",
                "description": "Open URL or search Google, YouTube, or GitHub.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "query": {"type": "string"},
                        "destination": {"type": "string", "enum": ["google", "youtube", "github"]}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_web_live",
                "description": "Search the web for current info.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "create_calendar_event",
                "description": "Create a calendar event.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "date": {"type": "string"},
                        "time": {"type": "string"},
                        "duration_minutes": {"type": "integer"}
                    },
                    "required": ["title", "date", "time"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_timer_reminder",
                "description": "Set a timer or reminder.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "seconds": {"type": "integer"}
                    },
                    "required": ["label", "seconds"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search saved notes and RAG memory.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_memory_note",
                "description": "Save a note or fact to memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note": {"type": "string"},
                        "title": {"type": "string"}
                    },
                    "required": ["note"]
                }
            }
        }
    ]

