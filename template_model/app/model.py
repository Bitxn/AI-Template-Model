"""
model.py
Analyzes a user prompt and returns the optimal project folder tree as JSON.
Currently powered by Gemini 1.5 Flash (free tier: 1500 req/day).
Phase 3: swap analyze_prompt() body for your fine-tuned Llama inference call.
"""

import os
import json
import re
import google.generativeai as genai
from typing import Any

# ── Gemini client (initialized once at im
# port time) ───────────────────────────

genai.configure(api_key="AIzaSyDjBIVWsRdgFZj46h9vpPTn1Z4Ue6uRzTE")

_gemini_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0.2,       # Low = deterministic, consistent JSON output
        "max_output_tokens": 4096,
        "top_p": 0.8,
    },
)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a software architect AI. Your ONLY job is to analyze a user's app prompt and return a JSON project folder tree.

You must decide:
1. STACK TYPE — one of: "frontend_only", "backend_only", "full_stack"
   - frontend_only: static sites, landing pages, portfolios, dashboards with no server logic, browser games, tools that use only external APIs directly
   - backend_only: APIs, data pipelines, scrapers, cron jobs, CLI tools, webhooks
   - full_stack: apps that need both a UI AND a backend (auth, database, file uploads, custom APIs, real-time features)

2. OPTIMIZATIONS — what to ADD or REMOVE from the base template:
   - Add `auth/` under backend/services if authentication is needed
   - Add `websocket.py` under backend/ if real-time/chat is needed
   - Add `workers/` under backend/ if background jobs/queues are needed
   - Add `tests/` at root if the prompt implies a production/enterprise app
   - Remove `db/` entirely if no persistent storage is needed (pure stateless API)
   - Remove `services/` if logic is trivially simple (single-file backend)
   - Add extra page files under src/pages/ based on the app's screens
   - Add `store/` under src/ if complex state management (Redux/Zustand) is needed
   - Add `types/` under src/ if TypeScript types will be complex

3. NAME THE PROJECT — derive a short snake_case project name from the prompt (e.g. "todo_app", "ecommerce_store")

You MUST respond with ONLY a valid JSON object. No markdown, no explanation, no backticks.

The JSON schema is:
{
  "stack_type": "frontend_only" | "backend_only" | "full_stack",
  "project_name": "string",
  "reasoning": "one sentence explaining why this stack was chosen",
  "optimizations_applied": ["list of changes made from base template"],
  "tree": { ...folder tree object... }
}

The tree object schema (recursive):
- Directory: { "type": "directory", "name": "string", "children": [...] }
- File:      { "type": "file",      "name": "string" }

Root of tree is always a directory named after project_name."""


# ── Base templates (mirrors your runtime_template exactly) ────────────────────

FULL_STACK_TEMPLATE: dict[str, Any] = {
    "type": "directory",
    "name": "project",
    "children": [
        {
            "type": "directory",
            "name": "backend",
            "children": [
                {"type": "directory", "name": "__pycache__", "children": []},
                {
                    "type": "directory",
                    "name": "api",
                    "children": [
                        {"type": "file", "name": "__init__.py"},
                        {"type": "file", "name": "routes.py"},
                    ],
                },
                {
                    "type": "directory",
                    "name": "db",
                    "children": [
                        {"type": "file", "name": "__init__.py"},
                        {"type": "file", "name": "models.py"},
                        {"type": "file", "name": "database.py"},
                    ],
                },
                {
                    "type": "directory",
                    "name": "services",
                    "children": [
                        {"type": "file", "name": "__init__.py"},
                        {"type": "file", "name": "service.py"},
                    ],
                },
                {"type": "file", "name": "app.db"},
                {"type": "file", "name": "main.py"},
                {"type": "file", "name": "runtime.db"},
                {"type": "file", "name": "utils.py"},
            ],
        },
        {"type": "directory", "name": "dist", "children": []},
        {
            "type": "directory",
            "name": "public",
            "children": [{"type": "file", "name": "favicon.ico"}],
        },
        {
            "type": "directory",
            "name": "src",
            "children": [
                {"type": "directory", "name": "components", "children": []},
                {
                    "type": "directory",
                    "name": "pages",
                    "children": [{"type": "file", "name": "index.tsx"}],
                },
                {"type": "directory", "name": "hooks", "children": []},
                {
                    "type": "directory",
                    "name": "lib",
                    "children": [{"type": "file", "name": "api.ts"}],
                },
                {"type": "file", "name": "App.tsx"},
                {"type": "file", "name": "main.tsx"},
                {"type": "file", "name": "index.css"},
            ],
        },
        {"type": "file", "name": ".env"},
        {"type": "file", "name": "index.html"},
        {"type": "file", "name": "manifest.json"},
        {"type": "file", "name": "package.json"},
        {"type": "file", "name": "postcss.config.js"},
        {"type": "file", "name": "tailwind.config.js"},
        {"type": "file", "name": "tsconfig.json"},
        {"type": "file", "name": "user_data.txt"},
    ],
}

FRONTEND_ONLY_TEMPLATE: dict[str, Any] = {
    "type": "directory",
    "name": "project",
    "children": [
        {"type": "directory", "name": "dist", "children": []},
        {
            "type": "directory",
            "name": "public",
            "children": [{"type": "file", "name": "favicon.ico"}],
        },
        {
            "type": "directory",
            "name": "src",
            "children": [
                {"type": "directory", "name": "components", "children": []},
                {
                    "type": "directory",
                    "name": "pages",
                    "children": [{"type": "file", "name": "index.tsx"}],
                },
                {"type": "directory", "name": "hooks", "children": []},
                {"type": "directory", "name": "lib", "children": []},
                {"type": "file", "name": "App.tsx"},
                {"type": "file", "name": "main.tsx"},
                {"type": "file", "name": "index.css"},
            ],
        },
        {"type": "file", "name": ".env"},
        {"type": "file", "name": "index.html"},
        {"type": "file", "name": "manifest.json"},
        {"type": "file", "name": "package.json"},
        {"type": "file", "name": "postcss.config.js"},
        {"type": "file", "name": "tailwind.config.js"},
        {"type": "file", "name": "tsconfig.json"},
    ],
}

BACKEND_ONLY_TEMPLATE: dict[str, Any] = {
    "type": "directory",
    "name": "project",
    "children": [
        {
            "type": "directory",
            "name": "backend",
            "children": [
                {"type": "directory", "name": "__pycache__", "children": []},
                {
                    "type": "directory",
                    "name": "api",
                    "children": [
                        {"type": "file", "name": "__init__.py"},
                        {"type": "file", "name": "routes.py"},
                    ],
                },
                {
                    "type": "directory",
                    "name": "db",
                    "children": [
                        {"type": "file", "name": "__init__.py"},
                        {"type": "file", "name": "models.py"},
                        {"type": "file", "name": "database.py"},
                    ],
                },
                {
                    "type": "directory",
                    "name": "services",
                    "children": [
                        {"type": "file", "name": "__init__.py"},
                        {"type": "file", "name": "service.py"},
                    ],
                },
                {"type": "file", "name": "app.db"},
                {"type": "file", "name": "main.py"},
                {"type": "file", "name": "utils.py"},
            ],
        },
        {"type": "file", "name": ".env"},
        {"type": "file", "name": "requirements.txt"},
        {"type": "file", "name": "user_data.txt"},
    ],
}


# ── Core function ─────────────────────────────────────────────────────────────

def analyze_prompt(user_prompt: str) -> dict[str, Any]:
    """
    Send the user prompt to Gemini and get back the optimized folder tree.
    Phase 3: replace this body with your fine-tuned Llama inference call.
    Returns a fully parsed dict ready to be returned as JSON.
    """
    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Generate the folder tree for this app: {user_prompt}"
    )

    response = _gemini_model.generate_content(full_prompt)
    raw_text = response.text.strip()

    # Strip any accidental markdown fences Gemini might add
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model returned invalid JSON: {e}\nRaw output: {raw_text}"
        ) from e

    # Validate required top-level keys
    required_keys = {"stack_type", "project_name", "reasoning", "optimizations_applied", "tree"}
    missing = required_keys - result.keys()
    if missing:
        raise ValueError(f"Model response missing keys: {missing}")

    return result