"""
Batch tab helper formatting utilities, status tables, and YOLO label conversions.
"""

from __future__ import annotations

import html
from typing import Dict, List, Tuple
from interface.state import _STATUS_PILL


def render_status_table(image_status: Dict[str, dict], order: List[str]) -> str:
    """Render the HTML batch progress status table."""
    rows = []
    for stem in order:
        st = image_status.get(stem)
        if not st:
            continue
        pill = _STATUS_PILL.get(st["state"], _STATUS_PILL["queued"])
        score = st.get("score")
        score_txt = f"{score}/10" if score is not None else "—"
        rounds_txt = str(st.get("rounds_done", 0))
        detail = st.get("detail", "") or ""
        name_esc = html.escape(st["name"])
        detail_short = html.escape(detail[:120])
        detail_attr = html.escape(detail)
        rows.append(
            f"<tr><td>{name_esc}</td><td>{pill}</td>"
            f"<td>{rounds_txt}</td><td>{score_txt}</td>"
            f'<td style="color:#7d8590;font-size:0.7rem" title="{detail_attr}">{detail_short}</td></tr>'
        )
    body = (
        "".join(rows)
        if rows
        else '<tr><td colspan="5" style="color:#7d8590;text-align:center;padding:1rem;">No images yet.</td></tr>'
    )
    return f"""
<div class="output-panel" style="margin-top:0.75rem">
  <div class="out-header"><div class="out-header-left">
    <span class="out-header-dot"></span><span class="out-header-title">Batch Status ({len(order)} images)</span>
  </div></div>
  <div style="max-height:260px; overflow-y:auto;">
  <table class="batch-status-table">
    <thead><tr>
      <th>Image</th><th>Status</th>
      <th>Rounds</th><th>Score</th>
      <th>Detail</th>
    </tr></thead>
    <tbody>{body}</tbody>
  </table>
  </div>
</div>"""


def detections_to_yolo(
    detections: list, categories: list, allow_dynamic_classes: bool = False
) -> Tuple[List[str], List[str]]:
    """Convert pipeline detections (label + bbox_2d in 0-1000 coords) to YOLO
    lines '<class_id> <xc> <yc> <w> <h>' normalized to [0,1]. When allow_dynamic_classes is True,
    novel classes discovered by the VLM receive incrementing class IDs."""
    cat_lower = {}
    for i, c in enumerate(categories):
        c = (c or "").strip().lower()
        if c:
            cat_lower.setdefault(c, len(cat_lower))

    lines = []
    unmapped = []
    for det in detections or []:
        label = (det.get("label") or "").strip()
        if not label or label.lower() in ("none", "background", "error"):
            continue
        cls_id = cat_lower.get(label.lower())
        if cls_id is None:
            if allow_dynamic_classes:
                cls_id = len(cat_lower)
                cat_lower[label.lower()] = cls_id
            else:
                unmapped.append(label)
                continue
        try:
            x1, y1, x2, y2 = (float(v) for v in det["bbox_2d"])
        except (TypeError, ValueError):
            unmapped.append(label)
            continue
        xc = (x1 + x2) / 2.0 / 1000.0
        yc = (y1 + y2) / 2.0 / 1000.0
        w = (x2 - x1) / 1000.0
        h = (y2 - y1) / 1000.0
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines, unmapped

