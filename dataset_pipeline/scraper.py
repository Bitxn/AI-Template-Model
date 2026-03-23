"""
scraper.py
Scrapes GitHub repos via REST API (no Selenium needed) and extracts
folder tree structures. Handles rate limiting, retries, and quality filtering.
"""

import os
import time
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"

SEARCH_TOPICS = [
    "fastapi react fullstack",
    "fastapi nextjs",
    "saas dashboard react",
    "react typescript app",
    "nextjs app router",
    "express react fullstack",
    "django react app",
    "flask react frontend",
    "nodejs typescript backend",
    "react vite tailwind app",
    "chat app websocket",
    "ecommerce fullstack",
    "crud app fastapi",
    "authentication jwt react",
    "todo app fullstack",
]

GOOD_FOLDERS = {
    "src", "backend", "frontend", "api", "app",
    "client", "server", "pages", "components",
}

BAD_KEYWORDS = {
    "tutorial", "course", "learn", "example", "demo",
    "boilerplate", "template", "starter", "workshop",
    "exercise", "practice", "awesome",
}


# ── GitHub client ─────────────────────────────────────────────────────────────

class GitHubClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def _get(self, url: str, params: dict = None) -> Optional[dict]:
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=15)

                if resp.status_code == 403:
                    reset_at = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                    wait = max(reset_at - int(time.time()), 5)
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                if resp.status_code in (404, 401, 422):
                    return None

                resp.raise_for_status()
                return resp.json()

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout attempt {attempt + 1}: {url}")
                time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error attempt {attempt + 1}: {e}")
                time.sleep(2 ** attempt)

        return None

    def search_repos(self, query: str, per_page: int = 30, page: int = 1) -> list[dict]:
        data = self._get(f"{GITHUB_API}/search/repositories", params={
            "q": query, "sort": "stars", "order": "desc",
            "per_page": per_page, "page": page,
        })
        return data.get("items", []) if data else []

    def get_tree(self, owner: str, repo: str, branch: str = "HEAD") -> Optional[list[dict]]:
        data = self._get(
            f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}",
            params={"recursive": "1"},
        )
        return data.get("tree", []) if data else None

    def get_readme(self, owner: str, repo: str) -> Optional[str]:
        import base64
        data = self._get(f"{GITHUB_API}/repos/{owner}/{repo}/readme")
        if not data:
            return None
        try:
            return base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")
        except Exception:
            return None

    def get_default_branch(self, owner: str, repo: str) -> str:
        data = self._get(f"{GITHUB_API}/repos/{owner}/{repo}")
        return data.get("default_branch", "main") if data else "main"


# ── Filters ───────────────────────────────────────────────────────────────────

def is_quality_repo(repo: dict) -> bool:
    name = repo.get("name", "").lower()
    description = (repo.get("description") or "").lower()
    combined = f"{name} {description}"

    if repo.get("fork"):
        return False
    if repo.get("stargazers_count", 0) < 30:
        return False
    if repo.get("size", 0) < 5:
        return False
    if any(bad in combined for bad in BAD_KEYWORDS):
        return False
    if not repo.get("description"):
        return False
    return True


def has_meaningful_structure(flat_tree: list[dict]) -> bool:
    dirs = {
        item["path"].split("/")[0].lower()
        for item in flat_tree
        if item.get("type") == "tree"
    }
    return bool(dirs & GOOD_FOLDERS)


# ── Tree builder ──────────────────────────────────────────────────────────────

def build_tree_json(flat_tree: list[dict], repo_name: str) -> dict:
    """Convert GitHub's flat file list into a nested JSON tree."""
    root = {"type": "directory", "name": repo_name, "children": []}

    for item in flat_tree:
        path = item.get("path", "")
        item_type = item.get("type", "blob")

        parts = path.split("/")
        if any(p.startswith(".") for p in parts):
            continue
        if path in ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock"):
            continue
        if "__pycache__" in parts:
            continue

        node = root
        for i, part in enumerate(parts):
            is_last = (i == len(parts) - 1)
            if is_last:
                if item_type == "blob":
                    node["children"].append({"type": "file", "name": part})
                else:
                    existing = next(
                        (c for c in node["children"]
                         if c["name"] == part and c["type"] == "directory"), None
                    )
                    if not existing:
                        new_dir = {"type": "directory", "name": part, "children": []}
                        node["children"].append(new_dir)
            else:
                existing = next(
                    (c for c in node["children"]
                     if c["name"] == part and c["type"] == "directory"), None
                )
                if existing:
                    node = existing
                else:
                    new_dir = {"type": "directory", "name": part, "children": []}
                    node["children"].append(new_dir)
                    node = new_dir

    return root


def infer_stack_type(flat_tree: list[dict]) -> str:
    paths = {item["path"].lower() for item in flat_tree}
    all_text = " ".join(paths)

    has_frontend = any(x in all_text for x in [
        "src/", "pages/", ".tsx", ".jsx", "package.json",
        "vite.config", "next.config", "tailwind",
    ])
    has_backend = any(x in all_text for x in [
        "main.py", "app.py", "requirements.txt", "fastapi",
        "django", "flask", "server.js", "index.js", "routes.py",
        "controllers/", "models.py", "schema.py",
    ])

    if has_frontend and has_backend:
        return "full_stack"
    elif has_frontend:
        return "frontend_only"
    elif has_backend:
        return "backend_only"
    return "full_stack"


# ── Main entry point ──────────────────────────────────────────────────────────

def scrape_repos(token: str, max_repos: int = 500) -> list[dict]:
    """
    Scrape GitHub repos and return list of dicts with tree + readme.
    Each dict: { repo_name, repo_url, description, stars, stack_type, readme, tree }
    """
    client = GitHubClient(token)
    results = []
    seen_repos: set[str] = set()

    logger.info(f"Scraping GitHub. Target: {max_repos} repos.")

    for topic in SEARCH_TOPICS:
        if len(results) >= max_repos:
            break

        logger.info(f"Topic: '{topic}'")

        for page in range(1, 4):
            if len(results) >= max_repos:
                break

            repos = client.search_repos(topic, per_page=30, page=page)
            if not repos:
                break

            for repo in repos:
                if len(results) >= max_repos:
                    break

                full_name = repo.get("full_name", "")
                if full_name in seen_repos:
                    continue
                seen_repos.add(full_name)

                if not is_quality_repo(repo):
                    continue

                owner, name = full_name.split("/")
                logger.info(f"Processing: {full_name} ({repo['stargazers_count']} ★)")

                branch = client.get_default_branch(owner, name)
                flat_tree = client.get_tree(owner, name, branch)
                if not flat_tree or not has_meaningful_structure(flat_tree):
                    continue

                readme = client.get_readme(owner, name)
                if not readme or len(readme.strip()) < 100:
                    continue

                results.append({
                    "repo_name": name,
                    "repo_url": repo.get("html_url", ""),
                    "description": repo.get("description", ""),
                    "stars": repo.get("stargazers_count", 0),
                    "stack_type": infer_stack_type(flat_tree),
                    "readme": readme[:3000].strip(),
                    "tree": build_tree_json(flat_tree, name),
                })

                logger.info(f"Collected [{len(results)}/{max_repos}]: {full_name}")
                time.sleep(0.5)

        time.sleep(1)

    logger.info(f"Done. Collected {len(results)} repos.")
    return results
