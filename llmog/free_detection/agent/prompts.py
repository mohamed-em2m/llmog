"""Prompt management, template loading, and DynaPrompt integration.

All prompt templates live in ``llmog/prompts/*.md``.  The template files are
the canonical source of truth — this module loads them (via DynaPrompt when
available, with a plain-file fallback) and exposes render helpers that inject
the per-call variables.

Public templates
----------------
DEFAULT_DETECTOR_TEMPLATE   — detector_agent.md
DEFAULT_JUDGE_TEMPLATE      — feedback_agent.md
DEFAULT_REALTIME_TEMPLATE   — realtime_detector.md
DEFAULT_CROP_VERIFY_TEMPLATE — crop_verify.md
DEFAULT_AUTO_LABEL_TEMPLATE — auto_label_classifier.md

Render helpers
--------------
render_detector_prompt(...)
render_judge_prompt(...)
get_realtime_prompt(...)
render_crop_verify_prompt(...)
render_auto_label_prompt(...)
"""

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


def _load_prompt_template(filename: str, fallback: str = "") -> str:
    """Read a prompt template from disk, returning *fallback* on any error."""
    path = PROMPTS_DIR / filename
    if path.is_file():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning(
                "Failed to load prompt from %s, using fallback: %s", path, exc
            )
    return fallback


def _dynaprompt_attr(name: str):
    """Return a DynaPrompt template attribute by name, or None if unavailable."""
    if _dynaprompt_instance is None:
        return None
    return getattr(_dynaprompt_instance, name, None)


# ---------------------------------------------------------------------------
# Template constants — loaded once at import time
# ---------------------------------------------------------------------------

DEFAULT_DETECTOR_TEMPLATE: str = (
    _dynaprompt_attr("detector_agent").text
    if _dynaprompt_attr("detector_agent") is not None
    else _load_prompt_template("detector_agent.md")
)

DEFAULT_JUDGE_TEMPLATE: str = (
    _dynaprompt_attr("feedback_agent").text
    if _dynaprompt_attr("feedback_agent") is not None
    else _load_prompt_template("feedback_agent.md")
)

DEFAULT_REALTIME_TEMPLATE: str = (
    _dynaprompt_attr("realtime_detector").text
    if _dynaprompt_attr("realtime_detector") is not None
    else _load_prompt_template("realtime_detector.md")
)

DEFAULT_CROP_VERIFY_TEMPLATE: str = (
    _dynaprompt_attr("crop_verify").text
    if _dynaprompt_attr("crop_verify") is not None
    else _load_prompt_template("crop_verify.md")
)

DEFAULT_AUTO_LABEL_TEMPLATE: str = (
    _dynaprompt_attr("auto_label_classifier").text
    if _dynaprompt_attr("auto_label_classifier") is not None
    else _load_prompt_template("auto_label_classifier.md")
)


# ---------------------------------------------------------------------------
# Jinja2 render helper
# ---------------------------------------------------------------------------

def _jinja_render(template: str, **kwargs) -> str:
    """Render *template* with Jinja2, falling back to str.format on failure."""
    try:
        return Template(template).render(**kwargs)
    except Exception:
        # Last-resort: simple str.replace for each variable.
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"{{{{ {key} }}}}", str(value))
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result


# ---------------------------------------------------------------------------
# Realtime prompt
# ---------------------------------------------------------------------------

def get_realtime_prompt(categories: list[str] | None = None) -> str:
    """Render the real-time free-detection prompt."""
    cats_str = categories_prompt_text(categories)

    dp = _dynaprompt_attr("realtime_detector")
    if dp is not None:
        try:
            return dp.render({"categories_list": cats_str}).text
        except Exception as exc:
            logger.warning(
                "DynaPrompt realtime rendering failed, falling back: %s", exc
            )

    return _jinja_render(DEFAULT_REALTIME_TEMPLATE, categories_list=cats_str)


# ---------------------------------------------------------------------------
# Categories text helper
# ---------------------------------------------------------------------------

_OPEN_WORLD_CATEGORIES_TEXT = (
    "**OPEN-WORLD DETECTION MODE** — no fixed category list.\n"
    "- Detect **every clearly visible, distinct object** in the image: people, animals, "
    "vehicles, furniture, sports equipment, buildings, signs, plants, food, tools, etc.\n"
    "- Give each detection a short, specific, lowercase label naming the object itself "
    "(e.g. `person`, `tennis_racket`, `airplane`, `scoreboard`, `parking_sign`).\n"
    "- Use concrete nouns — never generic labels like `object` or `thing`, and never the scene name."
)


def categories_prompt_text(categories: List[str] | None) -> str:
    """Render the *Categories to detect* block for a category list.

    ``["*"]`` (or any list containing ``"*"``) switches to open-world phrasing
    so the model is instructed to detect everything instead of matching a
    literal asterisk label.
    """
    cats = [c for c in (categories or []) if c]
    if not cats or "*" in cats:
        return _OPEN_WORLD_CATEGORIES_TEXT
    return ", ".join(cats)


# ---------------------------------------------------------------------------
# Detector prompt
# ---------------------------------------------------------------------------

def render_detector_prompt(
    template: str,
    categories: List[str],
    category_definitions: str,
    feedback: Optional[str] = None,
    actions: Optional[str] = None,
    som_proposals: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render detector prompt with feedback, structured actions, and SoM regions."""
    # --- Build feedback_block ---
    feedback_block = ""
    if feedback:
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
A separate quality-control reviewer inspected your last attempt on this image.{actions_section}

### Reviewer Feedback (context and reasoning)
{feedback}

### Your responsibilities
1. Apply every Required Action above EXACTLY as specified.
2. Keep all boxes from your previous detections that were NOT flagged.
3. Re-scan the image for any remaining missed objects.
4. Ensure no duplicates or false positives remain.
"""

    # --- Build SoM block ---
    som_block = ""
    if som_proposals:
        som_block = "\n\n## Candidate Regions (Set-of-Mark)\n"
        som_block += (
            "The image contains numbered candidate regions. If you detect an object "
            "that aligns with one of these regions, prefer its coordinates. "
            "Candidates (0–1000 scale):\n"
        )
        for prop in som_proposals:
            som_block += f"- Candidate #{prop['id']}: bbox_2d ≈ {prop['bbox_2d']}\n"
        som_block += "\nYou may refer to these candidates or output standard bounding boxes."

    # --- Render via DynaPrompt (if using the default template) ---
    cats_text = categories_prompt_text(categories)
    if template == DEFAULT_DETECTOR_TEMPLATE:
        dp = _dynaprompt_attr("detector_agent")
        if dp is not None:
            try:
                return dp.render(
                    {
                        "categories_list": cats_text,
                        "category_definitions": category_definitions + som_block,
                        "feedback_block": feedback_block,
                    }
                ).text
            except Exception as exc:
                logger.warning(
                    "DynaPrompt detector rendering failed, falling back: %s", exc
                )

    return _jinja_render(
        template,
        categories_list=cats_text,
        category_definitions=category_definitions + som_block,
        feedback_block=feedback_block,
    )


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

def render_judge_prompt(
    template: str,
    category_definitions: str,
    detections: List[Dict[str, Any]],
) -> str:
    """Render judge prompt with indexed detections."""
    indexed_detections = [
        {
            "box_index": f"Box #{idx}",
            "label": det.get("label"),
            "bbox_2d": det.get("bbox_2d"),
        }
        for idx, det in enumerate(detections, 1)
    ]
    dets_json = json.dumps(indexed_detections, indent=2)

    if template == DEFAULT_JUDGE_TEMPLATE:
        dp = _dynaprompt_attr("feedback_agent")
        if dp is not None:
            try:
                return dp.render(
                    {
                        "category_definitions": category_definitions,
                        "detections_json": dets_json,
                    }
                ).text
            except Exception as exc:
                logger.warning(
                    "DynaPrompt judge rendering failed, falling back: %s", exc
                )

    return _jinja_render(
        template,
        category_definitions=category_definitions,
        detections_json=dets_json,
    )


# ---------------------------------------------------------------------------
# Crop verification prompt
# ---------------------------------------------------------------------------

def render_crop_verify_prompt(label: str, template: Optional[str] = None) -> str:
    """Render the crop-verification YES/NO prompt for a given *label*.

    Args:
        label: The category label to check presence for in the crop.
        template: Optional override template string. If None, the default
            ``crop_verify.md`` template is used.

    Returns:
        The rendered prompt string ready to pass to the VLM.
    """
    t = template if template is not None else DEFAULT_CROP_VERIFY_TEMPLATE

    if template is None:
        dp = _dynaprompt_attr("crop_verify")
        if dp is not None:
            try:
                return dp.render({"label": label}).text
            except Exception as exc:
                logger.warning(
                    "DynaPrompt crop_verify rendering failed, falling back: %s", exc
                )

    return _jinja_render(t, label=label)


# ---------------------------------------------------------------------------
# Auto-label defect classification prompt
# ---------------------------------------------------------------------------

def render_auto_label_prompt(
    known_class_names: List[str],
    template: Optional[str] = None,
    class_mode: str = "hybrid",
    class_definitions: str = "",
) -> str:
    """Render the auto-annotation defect-classification prompt.

    Args:
        known_class_names: List of defect class names already discovered in
            this dataset run. The model will try to reuse one of these before
            inventing a new class name.
        template: Optional override template string. If None, the default
            ``auto_label_classifier.md`` template is used.
        class_mode: One of ``"strict"``, ``"hybrid"`` (default), or ``"free"``.
            Controls how tightly the model is bound to ``known_class_names``.
        class_definitions: Optional per-class description block (plain text)
            to inject into the prompt to help the model distinguish classes.

    Returns:
        The rendered prompt string ready to pass to the VLM.
    """
    mode = (class_mode or "hybrid").lower().strip()

    if known_class_names:
        known_str = ", ".join(known_class_names)
    else:
        known_str = "(none yet)"

    defs_block = ""
    if class_definitions and class_definitions.strip():
        defs_block = f"\n\nClass Definitions:\n{class_definitions.strip()}"

    if mode == "strict":
        if known_class_names:
            mode_instruction = (
                f"You MUST classify this crop into EXACTLY ONE of the following classes: {known_str}.\n"
                f"If the crop clearly does not match any class, respond with class='none' and confidence=1.\n"
                f"Do NOT invent new class names under any circumstances.{defs_block}"
            )
        else:
            mode_instruction = (
                "No target classes have been defined yet.\n"
                "Respond with class='unknown' and confidence=1."
            )
    elif mode == "free":
        mode_instruction = (
            "You are in OPEN-WORLD mode. Freely identify and name the most specific defect or object "
            "visible in this crop using a concise, lowercase, underscore-separated label (e.g. 'surface_scratch', "
            "'oil_stain', 'bent_component'). Do NOT limit yourself to any prior class list."
            + (defs_block if defs_block else "")
        )
    else:  # hybrid (default)
        if known_class_names:
            mode_instruction = (
                f"PRIORITY classes (reuse if the crop matches): {known_str}.\n"
                f"If this crop clearly matches one of the above, use that exact name.\n"
                f"If it is a genuinely different defect/object not covered by the list, "
                f"you MAY introduce a new concise lowercase class name.{defs_block}"
            )
        else:
            mode_instruction = (
                "No classes discovered yet — freely name the defect/object you see with "
                "a concise, lowercase, underscore-separated label."
                + (defs_block if defs_block else "")
            )

    # Try to inject the mode instruction into the template via DynaPrompt / Jinja2
    t = template if template is not None else DEFAULT_AUTO_LABEL_TEMPLATE

    if template is None:
        dp = _dynaprompt_attr("auto_label_classifier")
        if dp is not None:
            try:
                return dp.render(
                    {
                        "known_classes": known_str,
                        "mode_instruction": mode_instruction,
                        "class_mode": mode,
                    }
                ).text
            except Exception as exc:
                logger.warning(
                    "DynaPrompt auto_label_classifier rendering failed, falling back: %s",
                    exc,
                )

    # Fallback: append the mode instruction to the rendered template
    rendered = _jinja_render(t, known_classes=known_str)
    return rendered + "\n\n" + mode_instruction

