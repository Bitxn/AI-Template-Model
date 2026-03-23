# AI Model Template

Full project for building an indigenous AI model that takes a natural language
prompt and returns the optimal project folder structure as JSON.

---

## Project Structure

```
AI MODEL TEMPLATE/
│
├── template_model/          ← The live API (FastAPI + Claude now, Llama later)
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          ← FastAPI app + Lambda handler
│   │   └── model.py         ← Core logic (swap Claude → Llama here in Phase 3)
│   ├── tests/
│   │   └── test_api.py      ← Full test suite (mocked, no API cost)
│   ├── .env.example
│   ├── README.md
│   ├── requirements.txt
│   └── template.yaml        ← AWS SAM deployment config
│
└── dataset_pipeline/        ← Builds the training data for your own model
    ├── scraper.py            ← GitHub API scraper (no Selenium)
    ├── prompt_generator.py  ← Gemini Flash: README → training prompt
    ├── pipeline.py          ← Orchestrator (run this)
    ├── inspect_dataset.py   ← Stats and validation tool
    ├── requirements.txt
    └── .env.example
```

---

## Roadmap

| Phase | Status | What |
|-------|--------|------|
| Phase 1 | ✅ Done | Template Model API live (Claude-powered) |
| Phase 2 | 🔄 Now | Build training dataset via GitHub scraping |
| Phase 3 | ⏳ Next | Fine-tune Llama on the dataset, swap into API |

---

## Phase 2: Build the Dataset

```bash
cd dataset_pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Add GITHUB_TOKEN + GEMINI_API_KEY

# Run — start with 100 to test, then go to 500
python pipeline.py --max-repos 100

# Inspect what you collected
python inspect_dataset.py dataset.jsonl --show 5
```

You need **500+ pairs** before Phase 3 is worth doing.

---

## Phase 1: Template Model API (live now)

```bash
cd template_model
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Add ANTHROPIC_API_KEY

uvicorn app.main:app --reload --port 8001
# Docs: http://localhost:8001/docs

# Tests
pytest tests/ -v

# Deploy to AWS Lambda
sam build && sam deploy --guided \
  --parameter-overrides AnthropicApiKey=sk-ant-YOUR_KEY
```
