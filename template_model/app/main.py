"""
main.py
FastAPI application — Template Structure Model API
Runs locally via uvicorn OR on AWS Lambda via Mangum (same code, zero changes).
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.model import analyze_prompt

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Template Model API starting up")
    yield
    logger.info("Template Model API shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Template Structure Model API",
    description="Analyzes a user prompt and returns the optimal project folder tree as JSON.",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PromptRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Natural language description of the app to build.",
        examples=["Build a SaaS todo app with user auth and a REST API"],
    )


class TreeResponse(BaseModel):
    success: bool
    stack_type: str
    project_name: str
    reasoning: str
    optimizations_applied: list[str]
    tree: dict
    latency_ms: int


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    """Health check — used by ALB / Lambda warm-up pings."""
    return {"status": "ok", "service": "template-model"}


@app.post("/generate-tree", response_model=TreeResponse, tags=["model"])
async def generate_tree(body: PromptRequest):
    """
    Main endpoint. Accepts a prompt, returns the folder tree JSON.
    """
    start = time.perf_counter()
    logger.info(f"Prompt ({len(body.prompt)} chars): {body.prompt[:120]}...")

    try:
        result = analyze_prompt(body.prompt)
    except ValueError as e:
        logger.error(f"Model parsing error: {e}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal model error. Please retry.")

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        f"Done | stack={result['stack_type']} | project={result['project_name']} | {latency_ms}ms"
    )

    return TreeResponse(
        success=True,
        stack_type=result["stack_type"],
        project_name=result["project_name"],
        reasoning=result["reasoning"],
        optimizations_applied=result["optimizations_applied"],
        tree=result["tree"],
        latency_ms=latency_ms,
    )


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "detail": "Unexpected server error."},
    )


# ── AWS Lambda handler ────────────────────────────────────────────────────────

try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
    logger.info("Mangum handler registered — Lambda mode available")
except ImportError:
    handler = None
