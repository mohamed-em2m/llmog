"""Rebuild .checkpoint.json from staged batch_XXXX labels + output data.yaml.
Safe resume: per-image completion only, batches_done=[] so a resumed run
re-enters each batch but skips finished stems individually (works even if
--batch_size changed). Scans <output>/batches/ (current staging location)
plus legacy <output>/labels/batch_XXXX/ plus flat <output>/labels/*.txt.
Run: python rebuild_checkpoint.py
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import yaml

OUTPUT = Path(r"C:\Users\emam2\tech_projects\aggle\working\output")


def _resolve_paths(output):
    out = Path(output)
    return out, out / "labels", out / "data.yaml", out / ".checkpoint.json"


def pick_best(paths):
    """Prefer non-empty valid content, then newest mtime, then largest."""

    def score(p):
        try:
            st = p.stat()
            txt = p.read_text().strip()
        except Exception:
            return (0, 0, 0)
        nonempty = 1 if txt else 0
        return (nonempty, st.st_mtime, st.st_size)

    return max(paths, key=score)


def main(output=None):
    if output is None:
        ap = argparse.ArgumentParser(
            description="Rebuild output/.checkpoint.json from output labels + output data.yaml."
        )
        ap.add_argument(
            "--output",
            default=str(OUTPUT),
            help="Output folder holding labels/ + data.yaml (default: %(default)s).",
        )
        output = ap.parse_args().output
    _, LABELS, DATA_YAML, CKPT = _resolve_paths(output)
    assert LABELS.is_dir(), f"labels dir missing: {LABELS}"
    assert DATA_YAML.exists(), f"data.yaml missing: {DATA_YAML}"
    data = yaml.safe_load(open(DATA_YAML))
    names = data.get("names", []) or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    class_map = {str(n): i for i, n in enumerate(names)}
    print(f"class_map from output data.yaml ({len(class_map)}): {class_map}")

    by_stem = defaultdict(list)
    out = Path(output)
    seen: set[Path] = set()
    # Current staging location first, then legacy + flat labels.
    for base in (out / "batches", LABELS):
        if not base.is_dir():
            continue
        for p in base.rglob("*.txt"):
            if p not in seen:
                seen.add(p)
                by_stem[p.stem].append(p)

    print(f"total .txt files: {sum(len(v) for v in by_stem.values())}")
    print(f"unique stems: {len(by_stem)}")
    dups = {k: v for k, v in by_stem.items() if len(v) > 1}
    print(f"duplicated stems: {len(dups)}")
    diff = 0
    for k, v in dups.items():
        if len({x.read_text() for x in v}) > 1:
            diff += 1
    print(f"duplicated with DIFFERENT content: {diff} (kept best: non-empty > newest)")

    # best-copy report
    empty = sum(1 for v in by_stem.values() if not pick_best(v).read_text().strip())
    print(f"stems whose best copy is empty (valid YOLO empty): {empty}")
    print(f"stems with content: {len(by_stem) - empty}")

    completed = sorted(by_stem.keys())
    payload = {
        "completed_images": completed,
        "class_map": class_map,
        "batches_done": [],
    }
    # atomic write like CheckpointManager
    tmp = CKPT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(CKPT)
    print(f"Wrote {CKPT} with {len(completed)} completed_images, batches_done=[]")
    print(
        "Next: re-run the SAME command WITHOUT --no_auto_resume (auto_resume is ON by default)."
    )
    print(
        "Add --resume too for double safety. Keep --batch_size, --categories, --class_mode, --shuffle/--seed identical."
    )


if __name__ == "__main__":
    main()
