"""Prompt management, template loading, and DynaPrompt integration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from jinja2 import Template

logger = logging.getLogger("detection_pipeline")

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

try:
    from dynaprompt import DynaPrompt

    _dynaprompt_instance = DynaPrompt(settings_files=[str(PROMPTS_DIR)])
except Exception as _exc:
    logger.warning("Failed to initialize DynaPrompt from %s: %s", PROMPTS_DIR, _exc)
    _dynaprompt_instance = None


def _load_prompt_template(filename: str, fallback: str) -> str:
    path = PROMPTS_DIR / filename
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning(
                "Failed to load prompt from %s, using fallback: %s", path, exc
            )
    return fallback


DEFAULT_DETECTOR_TEMPLATE = (
    _dynaprompt_instance.detector_agent.text
    if _dynaprompt_instance is not None
    else ""
)
DEFAULT_JUDGE_TEMPLATE = (
    _dynaprompt_instance.feedback_agent.text
    if _dynaprompt_instance is not None
    else ""
)
DEFAULT_REALTIME_TEMPLATE = (
    _dynaprompt_instance.realtime_detector.text
    if _dynaprompt_instance is not None
    else ""
)


def get_realtime_prompt(categories: list[str] | None = None) -> str:
    """Render the real-time free-detection prompt."""
    cats_str = ", ".join(categories) if categories else "*"

    if _dynaprompt_instance is not None:
        try:
            return _dynaprompt_instance.realtime_detector.render(
                {"categories_list": cats_str}
            ).text
        except Exception as exc:
            logger.warning(
                "DynaPrompt realtime rendering failed, falling back: %s", exc
            )

    try:
        return Template(DEFAULT_REALTIME_TEMPLATE).render(categories_list=cats_str)
    except Exception:
        return DEFAULT_REALTIME_TEMPLATE.replace("{{ categories_list }}", cats_str)


def render_detector_prompt(
    template: str,
    categories: List[str],
    category_definitions: str,
    feedback: Optional[str] = None,
    actions: Optional[str] = None,
    som_proposals: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render detector prompt with feedback, structured actions, and candidate regions."""
    feedback_block = ""
    if feedback:
        prev_dets_section = ""
        actions_section = ""
        if actions and actions.strip().upper() != "NONE" and actions.strip():
            actions_section = f"""

### Required Actions (MANDATORY — apply these FIRST before re-scanning)
The reviewer identified these specific changes you MUST make to your previous detections:
```
{actions.strip()}
```
For each action line:
- `REMOVE #N` → do NOT include Box #N in your output.
- `RELABEL #N -> label` → keep Box #N's bounding box but change its label to the specified one.
- `MODIFY #N bbox -> [x1,y1,x2,y2]` → keep Box #N's label but replace its bbox_2d with the new coordinates.
- `ADD label at [x1,y1,x2,y2]` → add a new detection with the given label and coordinates."""

        feedback_block = f"""
## Correction Instructions from Quality Review
A separate quality-control reviewer inspected your last attempt on this image.{prev_dets_section}{actions_section}

### Reviewer Feedback (context and reasoning)
{feedback}

### Your responsibilities
1. Apply every Required Action above EXACTLY as specified.
2. Keep all boxes from your previous detections that were NOT flagged.
3. Re-scan the image for any remaining missed objects.
4. Ensure no duplicates or false positives remain.
"""
    som_block = ""
    if som_proposals:
        som_block = "\n\n## Candidate Regions (Set-of-Mark)\n"
        som_block += "The image contains numbered candidate regions. If you detect an object that aligns with one of these regions, you should prefer outputting its coordinates. Below is the list of candidates and their approximate coordinates on a 0-1000 scale:\n"
        for prop in som_proposals:
            som_block += f"- Candidate #{prop['id']}: label proposals around bbox_2d: {prop['bbox_2d']}\n"
        som_block += "\nYou can either refer to these candidates or output standard bounding boxes."

    if template == DEFAULT_DETECTOR_TEMPLATE and _dynaprompt_instance is not None:
        try:
            return _dynaprompt_instance.detector_agent.render(
                {
                    "categories_list": ", ".join(categories),
                    "category_definitions": category_definitions + som_block,
                    "feedback_block": feedback_block,
                }
            ).text
        except Exception as exc:
            logger.warning(
                "DynaPrompt detector rendering failed, falling back: %s", exc
            )

    try:
        return Template(template).render(
            categories_list=", ".join(categories),
            category_definitions=category_definitions + som_block,
            feedback_block=feedback_block,
        )
    except Exception:
        return template.format(
            categories_list=", ".join(categories),
            category_definitions=category_definitions + som_block,
            feedback_block=feedback_block,
        )


def render_judge_prompt(
    template: str,
    category_definitions: str,
    detections: List[Dict[str, Any]],
) -> str:
    """Render judge prompt formatted with indexed detections."""
    indexed_detections = []
    for idx, det in enumerate(detections, 1):
        indexed_detections.append(
            {
                "box_index": f"Box #{idx}",
                "label": det.get("label"),
                "bbox_2d": det.get("bbox_2d"),
            }
        )

    dets_json = json.dumps(indexed_detections, indent=2)

    if template == DEFAULT_JUDGE_TEMPLATE and _dynaprompt_instance is not None:
        try:
            return _dynaprompt_instance.feedback_agent.render(
                {
                    "category_definitions": category_definitions,
                    "detections_json": dets_json,
                }
            ).text
        except Exception as exc:
            logger.warning("DynaPrompt judge rendering failed, falling back: %s", exc)

    try:
        return Template(template).render(
            category_definitions=category_definitions,
            detections_json=dets_json,
        )
    except Exception:
        return template.format(
            category_definitions=category_definitions,
            detections_json=dets_json,
        )
