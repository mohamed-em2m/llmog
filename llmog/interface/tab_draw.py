"""
Draw & Recognize Reclassification Tab module.
Provides a dedicated large canvas interface for interactive region stroke drawing,
crop classification, and YOLO labeling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import gradio as gr

from interface.batch.reclassification import (
    classify_regions_gui,
    _RECLS_EMPTY_TABLE,
)

_DEFAULT_CANVAS_IMAGE = Path(__file__).resolve().parents[2] / "assets" / "image.png"


def build_draw_tab() -> Dict[str, Any]:
    """Build the dedicated Draw & Recognize tab with a full-height interactive canvas."""
    with gr.Row(equal_height=False):
        # ── Left / Main: Large Drawing Canvas ─────────────────────────────
        with gr.Column(scale=3, min_width=500):
            gr.HTML('<p class="section-label">🎨 Interactive Annotation Canvas</p>')
            recls_image_editor = gr.ImageEditor(
                label="Upload an image and draw a circle, rectangle, or outline in red over any object",
                type="pil",
                value=str(_DEFAULT_CANVAS_IMAGE) if _DEFAULT_CANVAS_IMAGE.is_file() else None,
                sources=["upload", "clipboard"],
                brush=gr.Brush(
                    default_size=4,
                    colors=["#ff0000", "#ff3333", "#00ff00", "#0088ff", "#ffff00", "#ff00ff", "#ffffff"],
                    color_mode="fixed",
                    default_color="#ff0000",
                ),
                eraser=gr.Eraser(default_size=12),
                layers=True,
                format="png",
                height=600,
            )

            with gr.Row(elem_classes=["btn-group"]):
                recls_run_btn = gr.Button(
                    "🔎  Recognize Drawn Regions",
                    variant="primary",
                    scale=2,
                    interactive=True,
                )

        # ── Right: Config & Recognition Results ───────────────────────────
        with gr.Column(scale=2, min_width=380):
            gr.HTML('<p class="section-label">⚙️ Classes &amp; Detection Settings</p>')

            recls_classes_input = gr.Textbox(
                label="Target Classes (comma-separated)",
                placeholder="hole, stain, tear, cut, knot, weaving_defect",
                value="hole, stain, tear, cut, knot, weaving_defect",
                lines=2,
            )

            recls_defs_input = gr.Textbox(
                label="Class Definitions / Prompt Context",
                lines=4,
                value=(
                    "- hole: missing fabric\n"
                    "- stain: discoloration only\n"
                    "- tear: frayed, uneven separation\n"
                    "- cut: clean cut\n"
                    "- knot: raised lump\n"
                    "- weaving_defect: uneven thread density"
                ),
            )

            recls_padding_slider = gr.Slider(
                label="Region Context Padding (%)",
                minimum=0,
                maximum=50,
                step=1,
                value=10,
                info="Extra visual context around each drawn region sent to the VLM.",
            )

            recls_status = gr.Markdown("**Status: Idle**")
            recls_annotated = gr.Image(
                label="Annotated Recognition Result",
                type="pil",
                interactive=False,
            )

            recls_results = gr.HTML(value=_RECLS_EMPTY_TABLE)

            with gr.Accordion("YOLO Labels (<class_id> <xc> <yc> <w> <h>)", open=False):
                recls_yolo = gr.Textbox(
                    lines=6,
                    interactive=False,
                    label="Copy these lines into the image's .txt label file",
                )

    return dict(
        recls_image_editor=recls_image_editor,
        recls_classes_input=recls_classes_input,
        recls_defs_input=recls_defs_input,
        recls_padding_slider=recls_padding_slider,
        recls_run_btn=recls_run_btn,
        recls_status=recls_status,
        recls_annotated=recls_annotated,
        recls_results=recls_results,
        recls_yolo=recls_yolo,
    )


def wire_draw_events(c_draw: Dict[str, Any], c_srv: Dict[str, Any], c_bat: Dict[str, Any]) -> None:
    """Wire interactive recognition events for the Draw & Recognize tab."""
    c_draw["recls_run_btn"].click(
        fn=classify_regions_gui,
        inputs=[
            c_draw["recls_image_editor"],
            c_draw["recls_classes_input"],
            c_draw["recls_defs_input"],
            c_draw["recls_padding_slider"],
            c_bat["use_external_api_chk"],
            c_bat["ext_api_url"],
            c_bat["ext_api_key"],
            c_bat["ext_model_name"],
            c_srv["server_port_input"],
        ],
        outputs=[
            c_draw["recls_status"],
            c_draw["recls_annotated"],
            c_draw["recls_results"],
            c_draw["recls_yolo"],
        ],
        concurrency_limit=1,
    )
