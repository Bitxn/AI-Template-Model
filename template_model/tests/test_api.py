"""
test_api.py
Full test suite for the Template Structure Model API.
All tests use mocks — no real API calls, no cost.

Run with:
    pytest tests/ -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.model import analyze_prompt

client = TestClient(app)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_mock_response(
    stack_type: str,
    project_name: str,
    optimizations: list[str] | None = None,
) -> dict:
    return {
        "stack_type": stack_type,
        "project_name": project_name,
        "reasoning": f"This is a {stack_type} app.",
        "optimizations_applied": optimizations or [],
        "tree": {
            "type": "directory",
            "name": project_name,
            "children": [
                {"type": "file", "name": "README.md"}
            ],
        },
    }


def mock_claude_message(content_text: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=content_text)]
    return msg


# ── Health ────────────────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Input validation ──────────────────────────────────────────────────────────

def test_empty_prompt_rejected():
    assert client.post("/generate-tree", json={"prompt": ""}).status_code == 422

def test_too_short_prompt_rejected():
    assert client.post("/generate-tree", json={"prompt": "hi"}).status_code == 422

def test_too_long_prompt_rejected():
    assert client.post("/generate-tree", json={"prompt": "x" * 2001}).status_code == 422

def test_missing_prompt_field():
    assert client.post("/generate-tree", json={}).status_code == 422


# ── Stack type detection ──────────────────────────────────────────────────────

@patch("app.model.client")
def test_frontend_only(mock_client):
    mock_client.messages.create.return_value = mock_claude_message(
        json.dumps(make_mock_response("frontend_only", "portfolio_site"))
    )
    resp = client.post("/generate-tree", json={"prompt": "Build a portfolio website with animations"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["stack_type"] == "frontend_only"
    assert data["latency_ms"] >= 0


@patch("app.model.client")
def test_backend_only(mock_client):
    mock_client.messages.create.return_value = mock_claude_message(
        json.dumps(make_mock_response("backend_only", "scraper_api"))
    )
    resp = client.post("/generate-tree", json={"prompt": "Build a web scraper API that collects product prices"})
    assert resp.status_code == 200
    assert resp.json()["stack_type"] == "backend_only"


@patch("app.model.client")
def test_full_stack(mock_client):
    mock_client.messages.create.return_value = mock_claude_message(
        json.dumps(make_mock_response(
            "full_stack", "chat_app",
            ["Added websocket.py for real-time messaging", "Added auth/ service"]
        ))
    )
    resp = client.post("/generate-tree", json={"prompt": "Build a real-time chat app with user auth"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["stack_type"] == "full_stack"
    assert len(data["optimizations_applied"]) == 2


# ── Error handling ────────────────────────────────────────────────────────────

@patch("app.model.client")
def test_bad_json_returns_502(mock_client):
    mock_client.messages.create.return_value = mock_claude_message("Not JSON at all!!")
    resp = client.post("/generate-tree", json={"prompt": "Build something cool please"})
    assert resp.status_code == 502


@patch("app.model.client")
def test_missing_keys_returns_502(mock_client):
    mock_client.messages.create.return_value = mock_claude_message(
        json.dumps({"stack_type": "full_stack"})  # missing required keys
    )
    resp = client.post("/generate-tree", json={"prompt": "Build something cool please"})
    assert resp.status_code == 502


# ── Markdown fence stripping ──────────────────────────────────────────────────

@patch("app.model.client")
def test_strips_markdown_fences(mock_client):
    """Model sometimes wraps output in ```json ... ``` — must still parse."""
    mock_response = make_mock_response("frontend_only", "clean_app")
    wrapped = f"```json\n{json.dumps(mock_response)}\n```"
    mock_client.messages.create.return_value = mock_claude_message(wrapped)
    result = analyze_prompt("Build a landing page")
    assert result["stack_type"] == "frontend_only"


# ── Response schema ───────────────────────────────────────────────────────────

@patch("app.model.client")
def test_response_has_all_fields(mock_client):
    mock_client.messages.create.return_value = mock_claude_message(
        json.dumps(make_mock_response("full_stack", "ecommerce_store", ["Added auth/"]))
    )
    resp = client.post("/generate-tree", json={"prompt": "Build an ecommerce store with payments"})
    data = resp.json()
    required = {"success", "stack_type", "project_name", "reasoning", "optimizations_applied", "tree", "latency_ms"}
    assert required.issubset(data.keys())
    assert isinstance(data["optimizations_applied"], list)
    assert isinstance(data["tree"], dict)
    assert data["tree"]["type"] == "directory"
