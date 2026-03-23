"""
prompt_generator.py
Uses Gemini Flash (free: 1500 req/day) to read a repo README
and generate a natural language training prompt.

To swap to Anthropic later, replace the generate_prompt_from_readme()
function body — the interface stays identical.
"""

import time
import logging
import json
import google.generativeai as genai

logger = logging.getLogger(__name__)


# ── Init ──────────────────────────────────────────────────────────────────────

def init_gemini(api_key: str):
    """Call once at startup."""
    genai.configure(api_key=api_key)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are a dataset labeler for an AI model that learns to generate project folder structures.

Your job: read a GitHub repo's README and description, then write ONE clear natural sentence
that describes what someone would type as a prompt to build this app from scratch.

Rules:
- Write as if a developer is asking an AI: "Build a...", "Create a...", "Make a..."
- Include KEY features that drive folder structure (auth, database, real-time, payments, etc.)
- Include tech stack ONLY if strongly implied (React, FastAPI, etc.)
- Keep it 1-2 sentences, 15-40 words
- Do NOT mention the repo name, author, or GitHub
- Output ONLY the prompt sentence, nothing else"""

GENERATION_CONFIG = {
    "temperature": 0.4,
    "max_output_tokens": 100,
    "top_p": 0.8,
}


# ── Rate limiter ──────────────────────────────────────────────────────────────

class RateLimiter:
    """Stay under Gemini free tier: 15 req/min, 1500/day."""
    def __init__(self, requests_per_minute: int = 12):
        self.min_interval = 60.0 / requests_per_minute
        self.last_call = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call
        wait_time = self.min_interval - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_call = time.time()


_rate_limiter = RateLimiter(requests_per_minute=12)


# ── Core function ─────────────────────────────────────────────────────────────

def generate_prompt_from_readme(
    readme: str,
    description: str,
    stack_type: str,
    repo_name: str,
) -> str | None:
    """
    Calls Gemini to generate a training prompt from a repo README.
    Returns the prompt string, or None on failure.
    """
    user_message = f"""Repo description: {description}
Stack type detected: {stack_type}

README (first 2000 chars):
{readme[:2000]}

Write the developer prompt for this app:"""

    _rate_limiter.wait()

    for attempt in range(3):
        try:
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config=GENERATION_CONFIG,
            )
            response = model.generate_content(user_message)

            if not response or not response.text:
                logger.warning(f"Empty Gemini response for {repo_name}")
                return None

            prompt_text = response.text.strip()

            if len(prompt_text) < 15 or len(prompt_text) > 500:
                logger.warning(f"Bad prompt length ({len(prompt_text)}) for {repo_name}")
                return None

            # Ensure it starts like a developer instruction
            first_word = prompt_text.split()[0].lower().rstrip(".,")
            if first_word not in {"build", "create", "make", "develop", "design", "implement", "write"}:
                prompt_text = f"Build a {prompt_text[0].lower()}{prompt_text[1:]}"

            return prompt_text

        except Exception as e:
            err = str(e).lower()
            if "quota" in err or "rate" in err or "429" in err:
                wait = 60 * (attempt + 1)
                logger.warning(f"Rate limit. Waiting {wait}s...")
                time.sleep(wait)
            elif "safety" in err or "blocked" in err:
                logger.warning(f"Gemini blocked {repo_name} (safety filter)")
                return None
            else:
                logger.error(f"Gemini error attempt {attempt + 1} for {repo_name}: {e}")
                time.sleep(2 ** attempt)

    return None


# ── Validation ────────────────────────────────────────────────────────────────

def validate_training_pair(prompt: str, tree: dict) -> bool:
    """Check both prompt and tree are meaningful before saving."""
    if not prompt or len(prompt.strip()) < 15:
        return False
    if tree.get("type") != "directory":
        return False
    children = tree.get("children", [])
    if len(children) < 2:
        return False
    if not any(c["type"] == "directory" for c in children):
        return False
    return True


def build_training_example(
    prompt: str,
    tree: dict,
    stack_type: str,
    repo_url: str,
    stars: int,
) -> dict:
    """
    Build the final training example in Alpaca instruction format.
    Compatible with unsloth, trl, axolotl, LLaMA-Factory out of the box.
    """
    return {
        "instruction": (
            "You are a software architect. Analyze the app description and return "
            "the optimal project folder structure as a JSON tree. Decide if the app "
            "needs frontend only, backend only, or full stack. Output only valid JSON."
        ),
        "input": prompt,
        "output": json.dumps(tree, separators=(",", ":")),
        "metadata": {
            "stack_type": stack_type,
            "source_repo": repo_url,
            "stars": stars,
        },
    }
