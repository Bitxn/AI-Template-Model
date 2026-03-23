"""
inspect_dataset.py
Inspect, validate and get stats on your collected dataset.

Usage:
    python inspect_dataset.py dataset.jsonl
    python inspect_dataset.py dataset.jsonl --show 5
"""

import json
import sys
import argparse
from collections import Counter
from pathlib import Path


def load_jsonl(file_path: str) -> list[dict]:
    records = []
    with open(file_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: Line {i} invalid JSON: {e}")
    return records


def count_nodes(tree: dict) -> tuple[int, int]:
    """Returns (files, dirs)."""
    if tree.get("type") == "file":
        return 1, 0
    files, dirs = 0, 1
    for child in tree.get("children", []):
        f, d = count_nodes(child)
        files += f
        dirs += d
    return files, dirs


def inspect(file_path: str, show_n: int = 0):
    path = Path(file_path)
    if not path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"\nInspecting: {file_path}")
    print("=" * 60)

    records = load_jsonl(file_path)
    if not records:
        print("No records found.")
        return

    print(f"Total training pairs: {len(records)}")

    # Stack distribution
    stack_counts = Counter(
        r.get("metadata", {}).get("stack_type", "unknown") for r in records
    )
    print("\nStack type distribution:")
    for stack, count in stack_counts.most_common():
        pct = count / len(records) * 100
        bar = "█" * int(pct / 2)
        print(f"  {stack:<20} {count:>4}  {bar} {pct:.1f}%")

    # Prompt length
    prompt_lens = [len(r.get("input", "")) for r in records]
    print(f"\nPrompt length — min: {min(prompt_lens)}  max: {max(prompt_lens)}  avg: {sum(prompt_lens)//len(prompt_lens)}")

    # Tree size
    file_counts = []
    for r in records:
        try:
            tree = json.loads(r.get("output", "{}"))
            f, _ = count_nodes(tree)
            file_counts.append(f)
        except Exception:
            pass

    if file_counts:
        print(f"Files per tree   — min: {min(file_counts)}  max: {max(file_counts)}  avg: {sum(file_counts)//len(file_counts)}")

    # Top starred repos
    sources = [
        (r.get("metadata", {}).get("stars", 0), r.get("metadata", {}).get("source_repo", ""))
        for r in records
    ]
    sources.sort(reverse=True)
    print("\nTop 5 highest-starred repos:")
    for stars, url in sources[:5]:
        print(f"  {stars:>6} ★  {url}")

    # Show examples
    if show_n > 0:
        print(f"\n{'='*60}")
        print(f"Showing {min(show_n, len(records))} examples:\n")
        for i, r in enumerate(records[:show_n]):
            print(f"--- Example {i+1} ---")
            print(f"Stack:  {r.get('metadata', {}).get('stack_type', 'unknown')}")
            print(f"Prompt: {r.get('input', '')}")
            try:
                tree = json.loads(r.get("output", "{}"))
                print("Tree root:")
                for child in tree.get("children", [])[:8]:
                    icon = "📁" if child["type"] == "directory" else "📄"
                    print(f"  {icon} {child['name']}")
                extras = len(tree.get("children", [])) - 8
                if extras > 0:
                    print(f"  ... +{extras} more")
            except Exception:
                print("  (could not parse tree)")
            print()

    # Readiness
    n = len(records)
    print("=" * 60)
    if n < 100:
        status = "❌ Too few — keep scraping"
    elif n < 500:
        status = "⚠️  Minimal — will work but may not generalize well"
    elif n < 1000:
        status = "✅ Good — solid fine-tune possible"
    else:
        status = "🚀 Excellent — great dataset"
    print(f"Readiness: {n} pairs → {status}\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect the collected dataset")
    parser.add_argument("file", help="Path to JSONL dataset file")
    parser.add_argument("--show", type=int, default=0, help="Examples to print")
    args = parser.parse_args()
    inspect(args.file, show_n=args.show)


if __name__ == "__main__":
    main()
