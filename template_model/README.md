# Template Model API

Analyzes a natural language prompt and returns the optimal project folder tree as JSON.
Built with FastAPI + Claude claude-sonnet-4-20250514. Deployable to AWS Lambda (serverless).

> **Phase 2**: Once `dataset_pipeline/` has generated 500+ training pairs, this API's
> `analyze_prompt()` in `app/model.py` will be swapped for a fine-tuned Llama model.

---

## API

### `POST /generate-tree`

```json
{ "prompt": "Build a real-time chat app with user auth and message history" }
```

**Response:**
```json
{
  "success": true,
  "stack_type": "full_stack",
  "project_name": "chat_app",
  "reasoning": "Needs React frontend and FastAPI backend for auth, WebSocket, and DB.",
  "optimizations_applied": ["Added websocket.py", "Added auth/ service"],
  "tree": { "type": "directory", "name": "chat_app", "children": [...] },
  "latency_ms": 3241
}
```

### `GET /health`
Returns `{"status": "ok"}`.

---

## Local Development

```bash
cd template_model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload --port 8001
```

Visit: http://localhost:8001/docs

## Tests

```bash
pytest tests/ -v
```

## AWS Deploy

```bash
sam build
sam deploy --guided \
  --parameter-overrides AnthropicApiKey=sk-ant-YOUR_KEY
```
