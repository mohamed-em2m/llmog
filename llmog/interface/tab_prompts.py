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

    with gr.Accordion("Prompt Templates — Detector & Judge (dropdown)", open=False):
        with gr.Row(equal_height=False, elem_classes=["draw-tab-row", "twin-screens-row"]):
            with gr.Column(scale=1, min_width=420):
                gr.HTML('<p class="section-label">Detector Prompt</p>')
                custom_det_prompt = gr.Textbox(
                    label="Detector Prompt Template",
                    lines=14,
                    value=DEFAULT_DETECTOR_TEMPLATE,
                )
                reset_det_btn = gr.Button(
                    "↩  Reset Detector Prompt to Default", size="sm", variant="secondary"
                )
            with gr.Column(scale=1, min_width=420):
                gr.HTML('<p class="section-label">Judge Prompt</p>')
                custom_jdg_prompt = gr.Textbox(
                    label="Judge Prompt Template",
                    lines=14,
                    value=DEFAULT_JUDGE_TEMPLATE,
                )
                reset_jdg_btn = gr.Button(
                    "↩  Reset Judge Prompt to Default", size="sm", variant="secondary"
                )

        reset_det_btn.click(
            fn=lambda: gr.update(value=DEFAULT_DETECTOR_TEMPLATE),
            inputs=None,
            outputs=[custom_det_prompt],
        )
        reset_jdg_btn.click(
            fn=lambda: gr.update(value=DEFAULT_JUDGE_TEMPLATE),
            inputs=None,
            outputs=[custom_jdg_prompt],
        )

    # Back-compat: hidden checkbox consumed by Batch wiring (always-on custom prompts)
    customize_prompts_chk = gr.Checkbox(label="Enable Custom Prompt Templates", value=True, visible=False)

    return dict(
        customize_prompts_chk=customize_prompts_chk,
        custom_det_prompt=custom_det_prompt,
        custom_jdg_prompt=custom_jdg_prompt,
    )
