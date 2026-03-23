"""
pipeline.py
Runs the full dataset generation pipeline end to end:
  1. Scrape GitHub repos
  2. Generate prompts via Gemini
  3. Save training pairs to dataset.jsonl

Usage:
    python pipeline.py
    python pipeline.py --max-repos 200 --output my_dataset.jsonl
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from scraper import scrape_repos
from prompt_generator import (
    init_gemini,
    generate_prompt_from_readme,
    validate_training_pair,
    build_training_example,
)

load_dotenv()


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_file: str = "pipeline.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a"),
        ],
    )

logger = logging.getLogger(__name__)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            with open(p) as f:
                data = json.load(f)
            logger.info(f"Resumed from checkpoint: {len(data.get('processed_repos', []))} done")
            return data
        except Exception as e:
            logger.warning(f"Checkpoint corrupted, starting fresh: {e}")
    return {"processed_repos": [], "pairs_saved": 0}


def save_checkpoint(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f)


# ── JSONL writer ──────────────────────────────────────────────────────────────

def append_to_jsonl(output_file: str, record: dict):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Stats ─────────────────────────────────────────────────────────────────────

class Stats:
    def __init__(self):
        self.scraped = 0
        self.prompt_fail = 0
        self.validation_fail = 0
        self.saved = 0
        self.start = time.time()

    def summary(self) -> str:
        elapsed = int(time.time() - self.start)
        m, s = divmod(elapsed, 60)
        return (
            f"\n{'='*50}\n"
            f"Pipeline complete in {m}m {s}s\n"
            f"  Repos scraped:        {self.scraped}\n"
            f"  Skipped (no prompt):  {self.prompt_fail}\n"
            f"  Skipped (validation): {self.validation_fail}\n"
            f"  Training pairs saved: {self.saved}\n"
            f"  Yield rate:           {self.saved / max(self.scraped, 1) * 100:.1f}%\n"
            f"{'='*50}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(
    github_token: str,
    gemini_api_key: str,
    max_repos: int = 500,
    output_file: str = "dataset.jsonl",
    checkpoint_file: str = "checkpoint.json",
):
    stats = Stats()
    logger.info("=" * 50)
    logger.info(f"Pipeline started — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Target: {max_repos} repos → {output_file}")
    logger.info("=" * 50)

    init_gemini(gemini_api_key)

    checkpoint = load_checkpoint(checkpoint_file)
    already_processed = set(checkpoint.get("processed_repos", []))

    # Step 1: Scrape
    logger.info("Step 1: Scraping GitHub repos...")
    repos = scrape_repos(github_token, max_repos=max_repos)
    stats.scraped = len(repos)
    logger.info(f"Scraped {len(repos)} repos")

    # Step 2+3: Generate prompts and save
    logger.info("Step 2: Generating prompts and saving pairs...")

    for i, repo in enumerate(repos):
        repo_url = repo["repo_url"]

        if repo_url in already_processed:
            continue

        logger.info(f"[{i+1}/{len(repos)}] {repo['repo_name']}")

        prompt = generate_prompt_from_readme(
            readme=repo["readme"],
            description=repo["description"],
            stack_type=repo["stack_type"],
            repo_name=repo["repo_name"],
        )

        if not prompt:
            stats.prompt_fail += 1
            already_processed.add(repo_url)
            continue

        if not validate_training_pair(prompt, repo["tree"]):
            stats.validation_fail += 1
            already_processed.add(repo_url)
            continue

        example = build_training_example(
            prompt=prompt,
            tree=repo["tree"],
            stack_type=repo["stack_type"],
            repo_url=repo_url,
            stars=repo["stars"],
        )

        append_to_jsonl(output_file, example)
        stats.saved += 1

        logger.info(f"  Pair #{stats.saved}: \"{prompt[:80]}\"")

        already_processed.add(repo_url)
        save_checkpoint(checkpoint_file, {
            "processed_repos": list(already_processed),
            "pairs_saved": stats.saved,
            "last_updated": datetime.now().isoformat(),
        })

    logger.info(stats.summary())
    if stats.saved > 0:
        logger.info(f"Dataset ready: {output_file}")
        logger.info("Run: python inspect_dataset.py dataset.jsonl")
    else:
        logger.warning("No pairs saved. Check your API keys and pipeline.log.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build prompt→tree training dataset")
    parser.add_argument("--max-repos", type=int, default=500)
    parser.add_argument("--output", type=str, default="dataset.jsonl")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.json")
    args = parser.parse_args()

    github_token = ""
    gemini_api_key =""

    if not github_token:
        raise SystemExit("ERROR: GITHUB_TOKEN not set in .env")
    if not gemini_api_key:
        raise SystemExit("ERROR: GEMINI_API_KEY not set in .env")

    setup_logging()
    run_pipeline(
        github_token=github_token,
        gemini_api_key=gemini_api_key,
        max_repos=args.max_repos,
        output_file=args.output,
        checkpoint_file=args.checkpoint,
    )


if __name__ == "__main__":
    main()
