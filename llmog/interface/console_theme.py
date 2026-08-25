"""
Red & Blue Retro Theme — Gradio theme tokens (5-color palette)
Pairs with console.css. One accent only: #D85040 (Red Glow).
Background #394A5B (Cold Steel), panels #3E5D7C (Ming Blue), muted #667E8B, hover #EA7555.

Palette:
  Rough Denim  #667E8B → muted/alt surfaces, disabled
  Ming Blue    #3E5D7C → cards / inputs / secondary
  Cold Steel   #394A5B → page background
  Red Glow     #D85040 → primary accent (buttons, active tab, slider)
  Burnt Sienna #EA7555 → hover accent
Derived:
  --ink  #26333F deep Cold Steel → borders + hard shadows
  --text #F2F6F8 cool near-white → text
"""

import gradio as gr

# Gradio's theme API is token-based; CSS in console.css overrides hard shadows / radius / hover.
# Primary hue = red (closest to #D85040), neutrals = slate (bluish grey close to #667E8B/#3E5D7C).

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[
        gr.themes.GoogleFont("Space Grotesk"),
        gr.themes.GoogleFont("JetBrains Mono"),
        "Space Grotesk",
        "JetBrains Mono",
        "Space Mono",
        "system-ui",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("JetBrains Mono"),
        gr.themes.GoogleFont("Space Mono"),
        "JetBrains Mono",
        "Space Mono",
        "monospace",
    ],
).set(
    # — page / surfaces —
    body_background_fill="#394A5B",
    body_background_fill_dark="#394A5B",
    body_text_color="#F2F6F8",
    body_text_color_dark="#F2F6F8",
    background_fill_primary="#3E5D7C",
    background_fill_primary_dark="#3E5D7C",
    background_fill_secondary="#394A5B",
    background_fill_secondary_dark="#394A5B",
    block_background_fill="#3E5D7C",
    block_background_fill_dark="#3E5D7C",
    # — borders: 3px solid ink everywhere (CSS enforces !important) —
    border_color_primary="#26333F",
    border_color_primary_dark="#26333F",
    border_color_accent="#26333F",
    # — Red Glow primary accent; Burnt Sienna hover —
    button_primary_background_fill="#D85040",
    button_primary_background_fill_dark="#D85040",
    button_primary_background_fill_hover="#EA7555",
    button_primary_background_fill_hover_dark="#EA7555",
    button_primary_text_color="#F2F6F8",
    button_primary_text_color_dark="#F2F6F8",
    button_primary_border_color="#26333F",
    button_primary_border_color_dark="#26333F",
    button_secondary_background_fill="#3E5D7C",
    button_secondary_background_fill_dark="#3E5D7C",
    button_secondary_background_fill_hover="#EA7555",
    button_secondary_background_fill_hover_dark="#EA7555",
    button_secondary_text_color="#F2F6F8",
    button_secondary_text_color_dark="#F2F6F8",
    button_secondary_border_color="#26333F",
    button_secondary_border_color_dark="#26333F",
    button_cancel_background_fill="#667E8B",
    button_cancel_background_fill_dark="#667E8B",
    button_cancel_background_fill_hover="#EA7555",
    button_cancel_background_fill_hover_dark="#EA7555",
    button_cancel_text_color="#F2F6F8",
    button_cancel_text_color_dark="#F2F6F8",
    button_cancel_border_color="#26333F",
    button_cancel_border_color_dark="#26333F",
    # — inputs: Ming Blue panel, ink border —
    input_background_fill="#3E5D7C",
    input_background_fill_dark="#3E5D7C",
    input_border_color="#26333F",
    input_border_color_dark="#26333F",
    input_border_color_focus="#26333F",
    input_border_color_focus_dark="#26333F",
    # — slider: Red Glow fill —
    slider_color="#D85040",
    slider_color_dark="#D85040",
    # — labels —
    block_label_text_color="#F2F6F8",
    block_label_text_color_dark="#F2F6F8",
    block_title_text_color="#F2F6F8",
    block_title_text_color_dark="#F2F6F8",
    block_label_background_fill="#3E5D7C",
    block_label_background_fill_dark="#3E5D7C",
    block_label_border_color="#26333F",
    block_label_border_color_dark="#26333F",
    # — radii: 0 everywhere —
    block_radius="0px",
    button_large_radius="0px",
    button_small_radius="0px",
    button_medium_radius="0px",
    input_radius="0px",
    checkbox_border_radius="0px",
    block_label_radius="0px",
    container_radius="0px",
    table_radius="0px",
    # — shadows: hard 5px, ink —
    block_shadow="5px 5px 0px #26333F",
    block_shadow_dark="5px 5px 0px #26333F",
    button_primary_shadow="5px 5px 0px #26333F",
    button_primary_shadow_dark="5px 5px 0px #26333F",
    button_secondary_shadow="5px 5px 0px #26333F",
    button_secondary_shadow_dark="5px 5px 0px #26333F",
    button_cancel_shadow="5px 5px 0px #26333F",
    button_cancel_shadow_dark="5px 5px 0px #26333F",
    input_shadow="5px 5px 0px #26333F",
    input_shadow_dark="5px 5px 0px #26333F",
    input_shadow_focus="5px 5px 0px #26333F",
    input_shadow_focus_dark="5px 5px 0px #26333F",
    stat_background_fill="#3E5D7C",
    stat_background_fill_dark="#3E5D7C",
    table_border_color="#26333F",
    table_border_color_dark="#26333F",
    table_even_background_fill="#3E5D7C",
    table_even_background_fill_dark="#3E5D7C",
    table_odd_background_fill="#394A5B",
    table_odd_background_fill_dark="#394A5B",
    code_background_fill="#3E5D7C",
    code_background_fill_dark="#3E5D7C",
)
