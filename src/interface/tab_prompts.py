"""
Prompts configuration tab UI.
"""

import gradio as gr
from free_detection.detection_pipeline import DEFAULT_DETECTOR_TEMPLATE, DEFAULT_JUDGE_TEMPLATE


def _build_prompts_tab():
    """Build the Prompts tab and return all interactive components."""

    gr.HTML('<p class="section-label">Prompt Engineering</p>')
    gr.Markdown(
        "Modify the custom instruction templates fed to the **Detector** and **Judge** agents.\n\n"
        "Available template variables: `{categories}`, `{category_definitions}`, `{feedback}`, `{detections_json}`"
    )
    gr.HTML(
        '<div class="input-hint">'
        "Template variables: "
        '<span class="hint-var">{categories}</span> '
        '<span class="hint-var">{category_definitions}</span> '
        '<span class="hint-var">{feedback}</span> '
        '<span class="hint-var">{detections_json}</span>'
        "</div>"
    )

    customize_prompts_chk = gr.Checkbox(
        label="Enable Custom Prompt Templates", value=False
    )

    with gr.Group(visible=False) as prompts_group:
        custom_det_prompt = gr.Textbox(
            label="Detector Prompt Template",
            lines=14,
            value=DEFAULT_DETECTOR_TEMPLATE,
        )
        custom_jdg_prompt = gr.Textbox(
            label="Judge Prompt Template",
            lines=14,
            value=DEFAULT_JUDGE_TEMPLATE,
        )

    customize_prompts_chk.change(
        lambda v: gr.update(visible=v),
        customize_prompts_chk,
        prompts_group,
    )

    return dict(
        customize_prompts_chk=customize_prompts_chk,
        custom_det_prompt=custom_det_prompt,
        custom_jdg_prompt=custom_jdg_prompt,
    )
