"""Render the whole-image classification prompt (DynaPrompt + fallback)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger("llmog.classify.prompt")

_PRESET_DEFS = {
    "fabric_defects": (
        "- hole: missing fabric or puncture\n"
        "- stain: discoloration or surface contaminant\n"
        "- tear: frayed, uneven physical separation\n"
        "- cut: clean sharp slice or incision\n"
        "- knot: raised thread lump or snarl\n"
        "- weaving_defect: uneven thread density or missing yarn"
    ),
    "coco": (
        "- person: human body\n"
        "- car: passenger automobile\n"
        "- bicycle: two-wheeled pedal bike\n"
        "- dog: canine domestic animal\n"
        "- cat: feline domestic animal\n"
        "- chair: seating furniture\n"
        "- bottle: liquid beverage container\n"
        "- laptop: portable notebook computer\n"
        "- cell_phone: handheld smartphone\n"
        "- book: bound printed volume"
    ),
    "road_traffic": (
        "- car: passenger sedan, coupe, or SUV\n"
        "- truck: heavy transport or cargo vehicle\n"
        "- pedestrian: person on foot\n"
        "- cyclist: person riding a bicycle\n"
        "- traffic_light: signal light lamp\n"
        "- traffic_sign: road regulatory or warning signboard\n"
        "- bus: public transit passenger bus\n"
        "- motorcycle: motorized two-wheeled vehicle"
    ),
    "retail_packaging": (
        "- box: cardboard or corrugated carton\n"
        "- barcode: 1D or 2D scanner code\n"
        "- product_label: brand packaging label\n"
        "- bottle: glass or plastic container\n"
        "- can: aluminum or tin can\n"
        "- pouch: flexible plastic packaging\n"
        "- blister_pack: clear molded plastic bubble packaging"
    ),
    "pcb_defects": (
        "- short_circuit: unintended electrical contact\n"
        "- missing_component: empty pad where SMD/component should be\n"
        "- solder_bridge: solder connecting adjacent pins\n"
        "- broken_trace: severed copper circuit trace\n"
        "- scratch: surface gouge across the solder mask\n"
        "- misalignment: component rotated or shifted off pad"
    ),
}


def resolve_class_definitions(class_definitions: str, preset: str | None) -> str:
    """Resolve --class_definitions (inline text or file path) + --preset."""
    raw = (class_definitions or "").strip()
    if raw and Path(raw).is_file():
        try:
            raw = Path(raw).read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("Could not read --class_definitions file %r: %s", raw, exc)
    if not raw and preset:
        raw = _PRESET_DEFS.get(preset, "")
    return raw


def parse_categories(categories: str) -> List[str]:
    return [c.strip() for c in (categories or "").split(",") if c.strip()]


def render_image_classify_prompt(
    categories: List[str],
    class_mode: str = "strict",
    class_definitions: str = "",
    classification_mode: str = "single",
    top_k: int = 3,
    multi_threshold: float = 50.0,
) -> str:
    """Render image_classifier.md via DynaPrompt, falling back to inline text."""
    mode = (class_mode or "strict").lower().strip()
    shape = (classification_mode or "single").lower().strip()
    cats = ", ".join(categories) if categories else "(none — open vocabulary)"
    defs = (class_definitions or "").strip()

    context = {
        "categories_list": cats,
        "class_definitions": defs,
        "class_mode": mode,
        "classification_mode": shape,
        "top_k": int(top_k or 3),
        "multi_threshold": float(multi_threshold),
    }
    try:
        from dynaprompt import get_prompt  # type: ignore

        dp = get_prompt("image_classifier")
        if dp is not None:
            try:
                rendered = dp.render(context)
                text = getattr(rendered, "text", rendered)
                if isinstance(text, str) and text.strip():
                    return text
            except TypeError:
                # Older dynaprompt render(**kwargs) signature
                rendered = dp.render(**context)
                text = getattr(rendered, "text", rendered)
                if isinstance(text, str) and text.strip():
                    return text
    except Exception as exc:
        logger.debug("DynaPrompt image_classifier render failed, fallback: %s", exc)

    try:
        template_path = (
            Path(__file__).resolve().parent.parent / "prompts" / "image_classifier.md"
        )
        template = template_path.read_text(encoding="utf-8")
        # Strip YAML frontmatter
        if template.startswith("---"):
            parts = template.split("---", 2)
            if len(parts) == 3:
                template = parts[2]
        from jinja2 import Template  # type: ignore

        return Template(template).render(**context)
    except Exception:
        pass

    # Minimal hardcoded fallback (no jinja/dynaprompt available)
    lines = [
        "You are an expert visual classifier. Analyze the FULL image.",
        f"Target classes: {cats}.",
    ]
    if defs:
        lines.append(f"Class definitions:\n{defs}")
    lines.append(f"Class expectation: {mode}. Output shape: {shape}.")
    if shape == "single":
        lines.append(
            'Respond with ONLY valid JSON: {"class": "<name>", '
            '"confidence": <0-100>, "reasoning": "<short>"}.'
        )
    elif shape == "multi":
        lines.append(
            'Respond with ONLY valid JSON: {"predictions": '
            '[{"class": "<name>", "confidence": <0-100>}], '
            '"reasoning": "<short>"}.'
        )
    else:
        lines.append(
            f"Respond with ONLY a JSON array of {int(top_k or 3)} objects: "
            '[{"class": "<name>", "confidence": <0-100>, '
            '"reasoning": "<short>"}].'
        )
    return "\n".join(lines)
