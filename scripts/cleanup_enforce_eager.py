#!/usr/bin/env python3
"""
Filter JSONL prediction caches by tokens-per-second threshold.

Use case:
- In ./outputs/zm60b/predictions/hf, each .jsonl line has field 'tok_per_s'.
- If tok_per_s > 40, it was generated under enforce_eager=False; otherwise True.
- This script removes lines generated under enforce_eager=True (tok_per_s <= threshold).

Example:
  python scripts/cleanup_enforce_eager.py \
    --root ./outputs/zm60b/predictions/hf \
    --threshold 40 \
    --backup
"""

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, List, Optional, Tuple


def discover_tok_per_s_path(root: str, sample_files: int = 3, sample_lines: int = 200) -> Optional[List[str]]:
    """Scan a few JSONL files to discover the most common path to 'tok_per_s'.

    Returns the path as a list of keys (e.g., ['metadata', 'perf', 'tok_per_s'])
    or None if not found.
    """
    def find_paths(obj: Dict, prefix: List[str]) -> Iterable[Tuple[str, ...]]:
        for k, v in obj.items():
            if k == 'tok_per_s':
                yield tuple(prefix + [k])
            if isinstance(v, dict):
                yield from find_paths(v, prefix + [k])

    paths = Counter()
    files_checked = 0
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith('.jsonl'):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, 'r', encoding='utf-8') as fin:
                    for i, line in enumerate(fin):
                        if i >= sample_lines:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(obj, dict):
                            for p in find_paths(obj, []):
                                paths[p] += 1
                files_checked += 1
                if files_checked >= sample_files:
                    raise StopIteration
            except StopIteration:
                break
            except Exception:
                continue
        if files_checked >= sample_files:
            break

    if not paths:
        return None
    # pick the most frequent path
    best_path, _ = paths.most_common(1)[0]
    return list(best_path)


def extract_tok_per_s(obj: Dict, path: Optional[List[str]]) -> Optional[float]:
    """Extract tok_per_s using a discovered path or fallback recursive search."""
    def get_by_path(d: Dict, p: List[str]) -> Optional[float]:
        cur: object = d
        for k in p:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        try:
            return float(cur)  # type: ignore[arg-type]
        except Exception:
            return None

    # Try explicit path first
    if path:
        val = get_by_path(obj, path)
        if val is not None:
            return val
    # Fallback: shallow and nested search for key
    stack = [obj]
    while stack:
        cur = stack.pop()
        if not isinstance(cur, dict):
            continue
        if 'tok_per_s' in cur:
            try:
                return float(cur['tok_per_s'])
            except Exception:
                return None
        for v in cur.values():
            if isinstance(v, dict):
                stack.append(v)
    return None


def process_file(path: str, threshold: float, make_backup: bool, tokps_path: Optional[List[str]]) -> Tuple[int, int]:
    """Filter a single JSONL file in-place by tok_per_s threshold.

    Returns (kept, removed) counts.
    """
    kept = 0
    removed = 0

    # Prepare backup if requested
    backup_path = f"{path}.bak"
    if make_backup:
        shutil.copy2(path, backup_path)

    # Write to temp file in the same directory, then replace (avoids cross-device issues)
    tmp_dir = os.path.dirname(os.path.abspath(path)) or "."
    with open(path, "r", encoding="utf-8") as fin, NamedTemporaryFile(
        "w", delete=False, encoding="utf-8", dir=tmp_dir
    ) as fout:
        tmp_path = fout.name
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tok_per_s = extract_tok_per_s(obj, tokps_path)
            except Exception:
                # If malformed, keep original line to avoid accidental data loss
                fout.write(line + "\n")
                kept += 1
                continue

            if tok_per_s is not None and tok_per_s > threshold:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
            else:
                removed += 1

    # Replace original atomically (same-filesystem); fallback to move on error
    try:
        os.replace(tmp_path, path)
    except OSError:
        # Cross-device or other edge cases: do a non-atomic move (uses top-level shutil)
        shutil.move(tmp_path, path)
    return kept, removed


def main():
    parser = argparse.ArgumentParser(description="Clean prediction caches by tok_per_s threshold.")
    parser.add_argument("--root", type=str, default="./outputs/zm60b/predictions/hf", help="Root dir of JSONL files")
    parser.add_argument("--threshold", type=float, default=40.0, help="Keep lines with tok_per_s > threshold")
    parser.add_argument("--backup", action="store_true", help="Make .bak backups alongside files before modifying")
    parser.add_argument("--dry-run", action="store_true", help="Only report counts; do not modify files")

    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"Root directory not found: {args.root}", file=sys.stderr)
        sys.exit(1)

    total_files = 0
    total_kept = 0
    total_removed = 0

    # discover tok_per_s path first
    discovered_path = discover_tok_per_s_path(args.root)
    if discovered_path:
        print(f"Discovered tok_per_s path: {'.'.join(discovered_path)}")
    else:
        print("Could not discover a consistent tok_per_s path; falling back to recursive search.")

    for dirpath, _, filenames in os.walk(args.root):
        for fname in filenames:
            if not fname.endswith(".jsonl"):
                continue
            fpath = os.path.join(dirpath, fname)
            total_files += 1

            if args.dry_run:
                # Simulate: read and count only
                kept = removed = 0
                with open(fpath, "r", encoding="utf-8") as fin:
                    for line in fin:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            tok_per_s = extract_tok_per_s(obj, discovered_path)
                        except Exception:
                            kept += 1
                            continue
                        if tok_per_s is not None and tok_per_s > args.threshold:
                            kept += 1
                        else:
                            removed += 1
                print(f"[DRY] {fpath}: keep={kept} remove={removed}")
                total_kept += kept
                total_removed += removed
            else:
                kept, removed = process_file(fpath, args.threshold, args.backup, discovered_path)
                print(f"{fpath}: keep={kept} remove={removed}")
                total_kept += kept
                total_removed += removed

    print(
        f"Done. Files={total_files}, kept={total_kept}, removed={total_removed}, threshold={args.threshold}")


if __name__ == "__main__":
    main()
