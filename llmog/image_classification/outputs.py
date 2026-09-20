"""CSV / YOLO-cls writers for whole-image classification results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

CSV_COLUMNS = [
    "filename",
    "path",
    "predicted_class",
    "confidence",
    "reasoning",
    "top_k_json",
    "all_scores_json",
    "status",
    "error",
]


def build_class_index(
    categories: List[str], all_predictions: List[List[Dict[str, Any]]]
) -> Dict[str, int]:
    """Order: user categories first, then newly discovered classes."""
    index: Dict[str, int] = {}
    for name in categories:
        key = name.strip().lower()
        if key and key not in index:
            index[key] = len(index)
    for preds in all_predictions:
        for p in preds:
            key = str(p.get("class") or "").strip().lower()
            if key and key not in ("none", "error") and key not in index:
                index[key] = len(index)
    return index


def write_predictions_csv(rows: List[Dict[str, Any]], csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
    return csv_path


def write_yolo_cls(
    rows: List[Dict[str, Any]],
    class_index: Dict[str, int],
    labels_dir: str | Path,
) -> List[Path]:
    """YOLO classification format: one `<class_id> <confidence0-1>` line per prediction."""
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        stem = Path(str(row.get("filename") or "image")).stem
        out = labels_dir / f"{stem}.txt"
        lines: List[str] = []
        try:
            preds = json.loads(row.get("all_scores_json") or "[]")
        except Exception:
            preds = []
        for p in preds:
            name = str(p.get("class") or "").strip().lower()
            if name not in class_index or name in ("none", "error"):
                continue
            conf = max(0.0, min(100.0, float(p.get("confidence", 0)))) / 100.0
            lines.append(f"{class_index[name]} {conf:.4f}")
        out.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
        written.append(out)
    return written


def write_classes_txt(class_index: Dict[str, int], path: str | Path) -> Path:
    path = Path(path)
    names = [name for name, _ in sorted(class_index.items(), key=lambda kv: kv[1])]
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
    return path


def write_summary_json(
    rows: List[Dict[str, Any]],
    class_index: Dict[str, int],
    path: str | Path,
) -> Path:
    path = Path(path)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    payload = {
        "total": len(rows),
        "ok": ok,
        "errors": len(rows) - ok,
        "classes": class_index,
        "results": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
