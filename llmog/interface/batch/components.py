"""
Batch tab Gradio layout components and UI builders.
"""

from __future__ import annotations

from typing import Dict, Any
import gradio as gr

from interface.state import (
    DEFAULT_CONCURRENCY,
    panel_header,
    _render_progress_bar,
    _section_title,
)
from interface.batch.runner import TASK_CHOICES, TASK_FREE_ANNOTATION
from interface.batch.helpers import render_status_table
from interface.batch.reclassification import _RECLS_EMPTY_TABLE, CATEGORY_PRESETS


def toggle_run_btn(is_running: bool):
    """Toggle Run and Cancel button states during pipeline execution."""
    return gr.update(interactive=not is_running), gr.update(interactive=is_running)


def toggle_external_api(use_external: bool):
    """Enable or disable local server vs external API inputs."""
    return (
        gr.update(interactive=not use_external),  # start_server_btn
        gr.update(interactive=not use_external),  # stop_server_btn
        gr.update(interactive=not use_external),  # server_preset
        gr.update(interactive=not use_external),  # server_backend
        gr.update(interactive=not use_external),  # server_model_input
        gr.update(interactive=not use_external),  # server_port_input
        gr.update(interactive=not use_external),  # server_thinking_chk
        gr.update(interactive=not use_external),  # server_mtp_chk
        gr.update(visible=use_external),          # ext_api_group
    )


def on_batch_preset_change(preset_name: str):
    """Populate batch categories and definitions when a domain preset is chosen."""
    preset = CATEGORY_PRESETS.get(preset_name, CATEGORY_PRESETS["Custom / Blank"])
    return (
        gr.update(value=preset["classes"]),
        gr.update(value=preset["defs"]),
    )


def on_batch_strategy_change(strategy: str):
    """Update batch inputs according to the selected category strategy."""
    if strategy == "free":
        return (
            gr.update(
                label="Domain / Focus Hint (Optional)",
                placeholder="e.g. Focus on industrial defects, biological specimens, vehicles... (or leave blank)",
                info="Free Mode: Agent autonomously detects all salient objects/anomalies.",
            ),
            gr.update(
                label="Domain Guidance / Prompt Context (Optional)",
                placeholder="Optional domain context or special instructions...",
                info="Optional domain guidance.",
            ),
            gr.update(visible=False),
        )
    elif strategy == "hybrid":
        return (
            gr.update(
                label="Priority Target Categories (comma-separated)",
                placeholder="e.g. hole, stain, tear, cut",
                info="Hybrid Mode: Target categories are prioritized; agent can discover new anomaly classes.",
            ),
            gr.update(
                label="Category Definitions & Novel Discovery Guidelines",
                placeholder="Definitions for priority categories...",
                info="Definitions for priority categories.",
            ),
            gr.update(visible=True),
        )
    else:  # strict
        return (
            gr.update(
                label="Target Categories (Strict - Comma Separated)",
                placeholder="hole, stain, tear, cut, knot, weaving_defect",
                info="Strict Mode: Agent is restricted strictly to listed categories.",
            ),
            gr.update(
                label="Category Definitions",
                placeholder="Write instructions for categories...",
                info="Category definitions.",
            ),
            gr.update(visible=True),
        )


def build_batch_tab() -> Dict[str, Any]:
    """Build the Batch Sandbox tab and return all interactive Gradio components."""

    with gr.Row(equal_height=False):
        # ── Left: Config ──────────────────────────────────────────────────
        with gr.Column(scale=2, min_width=400):
            gr.HTML('<p class="section-label">📁 Batch Detection Inputs</p>')

            input_images = gr.File(
                file_count="multiple",
                file_types=["image"],
                label="Upload Source Image(s)",
            )

            category_strategy = gr.Radio(
                label="🎯 Category Detection Strategy",
                choices=[
                    ("🔒 Strict (Closed-Set)", "strict"),
                    ("🔀 Hybrid (Extendable)", "hybrid"),
                    ("🌐 Free (Open-World)", "free"),
                ],
                value="strict",
                info="Strict: Only detects listed categories. Hybrid: Prioritizes target categories + discovers new. Free: Detects all salient objects freely.",
            )

            category_preset_dropdown = gr.Dropdown(
                label="📋 Category Domain Presets",
                choices=list(CATEGORY_PRESETS.keys()),
                value="Fabric & Surface Defects",
                info="Quickly load target categories & distinguishing definitions.",
            )

            categories_input = gr.Textbox(
                label="Target Categories (Strict - Comma Separated)",
                placeholder="hole, stain, tear, cut, knot, weaving_defect",
                value="hole, stain, tear, cut, knot, weaving_defect",
                info="Strict Mode: Agent is restricted strictly to listed categories.",
            )
            category_defs_input = gr.Textbox(
                label="Category Definitions",
                placeholder="Write instructions for categories...",
                lines=4,
                value=CATEGORY_PRESETS["Fabric & Surface Defects"]["defs"],
            )

            with gr.Accordion("Pipeline Parameters", open=False) as rounds_accordion:
                rounds_slider = gr.Slider(
                    label="Optimization Max Rounds",
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=1,
                )
                score_threshold_slider = gr.Slider(
                    label="Stop Score Threshold (0-10)",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=8,
                )
                det_temp_slider = gr.Slider(
                    label="Detector Temperature",
                    minimum=0.0,
                    maximum=1.5,
                    step=0.05,
                    value=0.9,
                )
                jdg_temp_slider = gr.Slider(
                    label="Judge Temperature",
                    minimum=0.0,
                    maximum=1.5,
                    step=0.05,
                    value=0.2,
                )

            with gr.Accordion(
                "Image Preprocessing & Augmentation", open=False
            ) as prep_accordion:
                prep_enabled_chk = gr.Checkbox(
                    label="Enable Preprocessing",
                    value=False,
                    info="Master toggle for all preprocessing steps below.",
                )

                with gr.Group(visible=False) as prep_options_group:
                    gr.HTML(_section_title("📐", "Resolution & Padding"))
                    prep_short_edge_slider = gr.Slider(
                        label="Target Short Edge (px)",
                        minimum=512,
                        maximum=2048,
                        step=128,
                        value=1024,
                        info="Upscale short edge to at least this value.",
                    )
                    prep_pad_square_chk = gr.Checkbox(
                        label="Pad to Square",
                        value=False,
                        info="Pad with neutral gray to maintain aspect ratio on square inputs.",
                    )

                    gr.HTML(_section_title("✂️", "Custom Resize"))
                    prep_custom_resize_chk = gr.Checkbox(
                        label="Enable Custom Resize (override short edge)",
                        value=False,
                        info="Resize all images to exact width × height. Overrides the short-edge target.",
                    )
                    with gr.Row(visible=False) as prep_custom_resize_row:
                        prep_custom_resize_width = gr.Number(
                            label="Target Width (px)",
                            value=1024,
                            precision=0,
                        )
                        prep_custom_resize_height = gr.Number(
                            label="Target Height (px)",
                            value=1024,
                            precision=0,
                        )

                    gr.HTML(_section_title("🎨", "Contrast & Color"))
                    prep_contrast_dropdown = gr.Dropdown(
                        label="Contrast Correction Method",
                        choices=["none", "clahe", "autocontrast"],
                        value="clahe",
                    )
                    prep_gamma_slider = gr.Slider(
                        label="Gamma Correction",
                        minimum=0.5,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                    )
                    prep_wb_chk = gr.Checkbox(
                        label="Gray World White Balance Correction",
                        value=False,
                    )

                    gr.HTML(_section_title("🔇", "Noise & Sharpness"))
                    prep_denoise_dropdown = gr.Dropdown(
                        label="Denoising Filter",
                        choices=["none", "bilateral", "nlm"],
                        value="none",
                    )
                    prep_sharpen_chk = gr.Checkbox(
                        label="Apply Unsharp Mask (Sharpen)", value=False
                    )

                    gr.HTML(_section_title("🔲", "Coordinate Grid Overlay"))
                    prep_grid_dropdown = gr.Dropdown(
                        label="Grid Style",
                        choices=["Standard Red", "transparent", "fine", "none"],
                        value="Standard Red",
                        info="Select standard, semi-transparent, fine 10×10 grid, or disable.",
                    )
                    prep_grid_step_slider = gr.Slider(
                        label="Grid Step Size (px)",
                        minimum=20,
                        maximum=500,
                        step=10,
                        value=250,
                        info="Distance between grid lines.",
                    )
                    prep_grid_line_width_slider = gr.Slider(
                        label="Grid Line Thickness (px)",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=1,
                    )
                    prep_grid_font_size_slider = gr.Slider(
                        label="Grid Label Font Size (0 = Auto)",
                        minimum=0,
                        maximum=48,
                        step=1,
                        value=0,
                    )
                    with gr.Row():
                        prep_grid_line_color_dropdown = gr.Dropdown(
                            label="Grid Line Color",
                            choices=[
                                "red",
                                "blue",
                                "green",
                                "white",
                                "black",
                                "yellow",
                                "cyan",
                                "magenta",
                                "custom",
                            ],
                            value="red",
                        )
                        prep_grid_line_color_custom = gr.Textbox(
                            label="Custom Line Color (Hex/Name)",
                            value="red",
                            visible=False,
                        )
                    with gr.Row():
                        prep_grid_text_color_dropdown = gr.Dropdown(
                            label="Grid Text Color",
                            choices=[
                                "white",
                                "black",
                                "red",
                                "blue",
                                "green",
                                "yellow",
                                "cyan",
                                "magenta",
                                "custom",
                            ],
                            value="white",
                        )
                        prep_grid_text_color_custom = gr.Textbox(
                            label="Custom Text Color (Hex/Name)",
                            value="white",
                            visible=False,
                        )
                    with gr.Row():
                        prep_grid_backing_color_dropdown = gr.Dropdown(
                            label="Grid Text Backing Color",
                            choices=[
                                "black",
                                "none",
                                "white",
                                "red",
                                "blue",
                                "green",
                                "custom",
                            ],
                            value="black",
                        )
                        prep_grid_backing_color_custom = gr.Textbox(
                            label="Custom Backing (Hex/Name)",
                            value="black",
                            visible=False,
                        )

                    gr.HTML(_section_title("🎯", "Visual Prompting (SoM)"))
                    prep_som_chk = gr.Checkbox(
                        label="Enable Set-of-Mark (SoM) Prompting",
                        value=False,
                        info="Detect candidate regions and overlay numbered circles as hints.",
                    )

                    gr.HTML(_section_title("🧩", "Tiling (Small Objects)"))
                    prep_tiling_chk = gr.Checkbox(
                        label="Enable Image Tiling",
                        value=False,
                        info="Split image into overlapping tiles, detect independently, and merge via NMS.",
                    )
                    prep_tile_size_slider = gr.Slider(
                        label="Tile Size (px)",
                        minimum=256,
                        maximum=1024,
                        step=128,
                        value=512,
                    )
                    prep_tile_overlap_slider = gr.Slider(
                        label="Tile Overlap (%)",
                        minimum=0,
                        maximum=50,
                        step=5,
                        value=20,
                    )

                    gr.HTML(_section_title("🔍", "Multi-Pass Crop & Verify"))
                    prep_cv_chk = gr.Checkbox(
                        label="Enable Crop & Verify Validation",
                        value=False,
                        info="Perform a second VLM validation pass on cropped detections.",
                    )
                    prep_cv_padding_slider = gr.Slider(
                        label="Crop Context Padding (%)",
                        minimum=0,
                        maximum=50,
                        step=5,
                        value=15,
                    )

                    gr.HTML(_section_title("📡", "VLM Processor Pixel Bounds"))
                    prep_send_pixel_bounds_chk = gr.Checkbox(
                        label="Send Pixel Bounds in API Request",
                        value=False,
                        info="Pass min_pixels/max_pixels in extra_body (Qwen-VL / vLLM backends).",
                    )
                    with gr.Row(visible=False) as prep_pixel_bounds_row:
                        prep_min_pixels_num = gr.Number(
                            label="min_pixels",
                            value=200704,
                            precision=0,
                            info="Default: 256×28×28",
                        )
                        prep_max_pixels_num = gr.Number(
                            label="max_pixels",
                            value=4194304,
                            precision=0,
                            info="Default: 2048×2048",
                        )

            with gr.Accordion("External API (Optional)", open=False) as ext_api_group:
                use_external_api_chk = gr.Checkbox(
                    label="Use External API instead of Local Server",
                    value=False,
                )
                ext_api_url = gr.Textbox(
                    label="Base URL", value="https://api.openai.com/v1"
                )
                ext_api_key = gr.Textbox(
                    label="API Key",
                    placeholder="sk-...",
                    value="",
                    type="password",
                )
                ext_model_name = gr.Textbox(label="Model Name", value="gpt-4o")

            with gr.Accordion("Advanced Settings", open=False) as advanced_accordion:
                concurrency_slider = gr.Slider(
                    label="Concurrent Images",
                    info=(
                        "Images processed in parallel. With a single-slot local server, "
                        "high values just queue at the server. Set 8–32 for external APIs "
                        "or multi-slot servers."
                    ),
                    minimum=1,
                    maximum=64,
                    step=1,
                    value=DEFAULT_CONCURRENCY,
                )

            with gr.Row(elem_classes=["btn-group"]):
                run_btn = gr.Button(
                    "▶  Run Batch Pipeline",
                    variant="primary",
                    interactive=True,
                    scale=3,
                )
                stop_run_btn = gr.Button(
                    "⏹  Cancel",
                    variant="secondary",
                    size="sm",
                    interactive=False,
                    scale=1,
                )

        # ── Right: Results ────────────────────────────────────────────────
        with gr.Column(scale=3, min_width=600):
            gr.HTML('<p class="section-label">Results</p>')

            with gr.Group():
                pipeline_status = gr.Markdown("**Status: Idle**")
                progress_html = gr.HTML(value=_render_progress_bar(0, "Idle"))

            batch_status_table = gr.HTML(value=render_status_table({}, []))
            download_results_box = gr.File(
                label="📥 Download Processed Results (.zip)",
                interactive=False,
            )

            with gr.Tabs() as explorer_tabs:
                with gr.TabItem("🖼️ Batch Explorer"):
                    with gr.Row():
                        explorer_image_select = gr.Dropdown(
                            label="Select Image",
                            choices=[],
                            interactive=True,
                            scale=2,
                        )
                        explorer_round_select = gr.Dropdown(
                            label="Select Round",
                            choices=[],
                            interactive=True,
                            scale=2,
                        )
                        round_score_display = gr.HTML(
                            value='<span class="score-badge">Score: -/10</span>',
                            elem_classes="score-display",
                            scale=1,
                        )

                    with gr.Row():
                        show_grid_chk = gr.Checkbox(
                            label="Show 0-1000 coordinate grid", value=True
                        )

                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            gr.HTML('<div class="img-viewer-wrap">')
                            source_image_viewer = gr.Image(
                                label="Source Image", type="pil"
                            )
                            gr.HTML("</div>")
                        with gr.Column(scale=1):
                            gr.HTML('<div class="img-viewer-wrap">')
                            best_annotated_viewer = gr.Image(
                                label="Annotated Image", type="pil"
                            )
                            gr.HTML("</div>")

                    round_feedback_display = gr.Textbox(
                        label="Judge's Feedback", lines=4, interactive=False
                    )

                    with gr.Accordion("Raw Response Details", open=False):
                        round_parse_error_display = gr.Textbox(
                            label="Parsing Errors", interactive=False
                        )
                        round_raw_response_display = gr.Textbox(
                            label="Raw Detector Text Response",
                            lines=6,
                            interactive=False,
                        )

                    with gr.Accordion("Batch Logs", open=False):
                        pipeline_logs_viewer = gr.Textbox(
                            lines=12,
                            max_lines=24,
                            interactive=False,
                            show_label=False,
                            container=False,
                        )

                with gr.TabItem("📄 Detections JSON"):
                    with gr.Group(elem_classes=["json-panel"]):
                        gr.HTML(
                            '<div class="json-panel-hdr"><span class="dot-amber"></span>'
                            "Detections (JSON List)</div>"
                        )
                        with gr.Group(elem_classes=["json-panel-body"]):
                            detections_json_box = gr.Code(
                                language="json",
                                show_label=False,
                                value="[]",
                            )

    return dict(
        explorer_tabs=explorer_tabs,
        rounds_accordion=rounds_accordion,
        prep_accordion=prep_accordion,
        advanced_accordion=advanced_accordion,
        input_images=input_images,
        category_strategy=category_strategy,
        category_preset_dropdown=category_preset_dropdown,
        categories_input=categories_input,
        category_defs_input=category_defs_input,
        rounds_slider=rounds_slider,
        score_threshold_slider=score_threshold_slider,
        det_temp_slider=det_temp_slider,
        jdg_temp_slider=jdg_temp_slider,
        prep_enabled_chk=prep_enabled_chk,
        prep_options_group=prep_options_group,
        prep_short_edge_slider=prep_short_edge_slider,
        prep_pad_square_chk=prep_pad_square_chk,
        prep_custom_resize_chk=prep_custom_resize_chk,
        prep_custom_resize_row=prep_custom_resize_row,
        prep_custom_resize_width=prep_custom_resize_width,
        prep_custom_resize_height=prep_custom_resize_height,
        prep_contrast_dropdown=prep_contrast_dropdown,
        prep_gamma_slider=prep_gamma_slider,
        prep_wb_chk=prep_wb_chk,
        prep_denoise_dropdown=prep_denoise_dropdown,
        prep_sharpen_chk=prep_sharpen_chk,
        prep_grid_dropdown=prep_grid_dropdown,
        prep_grid_step_slider=prep_grid_step_slider,
        prep_grid_line_width_slider=prep_grid_line_width_slider,
        prep_grid_font_size_slider=prep_grid_font_size_slider,
        prep_grid_line_color_dropdown=prep_grid_line_color_dropdown,
        prep_grid_line_color_custom=prep_grid_line_color_custom,
        prep_grid_text_color_dropdown=prep_grid_text_color_dropdown,
        prep_grid_text_color_custom=prep_grid_text_color_custom,
        prep_grid_backing_color_dropdown=prep_grid_backing_color_dropdown,
        prep_grid_backing_color_custom=prep_grid_backing_color_custom,
        prep_som_chk=prep_som_chk,
        prep_tiling_chk=prep_tiling_chk,
        prep_tile_size_slider=prep_tile_size_slider,
        prep_tile_overlap_slider=prep_tile_overlap_slider,
        prep_cv_chk=prep_cv_chk,
        prep_cv_padding_slider=prep_cv_padding_slider,
        prep_send_pixel_bounds_chk=prep_send_pixel_bounds_chk,
        prep_pixel_bounds_row=prep_pixel_bounds_row,
        prep_min_pixels_num=prep_min_pixels_num,
        prep_max_pixels_num=prep_max_pixels_num,
        ext_api_group=ext_api_group,
        use_external_api_chk=use_external_api_chk,
        ext_api_url=ext_api_url,
        ext_api_key=ext_api_key,
        ext_model_name=ext_model_name,
        concurrency_slider=concurrency_slider,
        run_btn=run_btn,
        stop_run_btn=stop_run_btn,
        pipeline_status=pipeline_status,
        progress_html=progress_html,
        batch_status_table=batch_status_table,
        download_results_box=download_results_box,
        explorer_image_select=explorer_image_select,
        explorer_round_select=explorer_round_select,
        round_score_display=round_score_display,
        show_grid_chk=show_grid_chk,
        source_image_viewer=source_image_viewer,
        best_annotated_viewer=best_annotated_viewer,
        round_feedback_display=round_feedback_display,
        round_parse_error_display=round_parse_error_display,
        round_raw_response_display=round_raw_response_display,
        detections_json_box=detections_json_box,
        pipeline_logs_viewer=pipeline_logs_viewer,
    )
