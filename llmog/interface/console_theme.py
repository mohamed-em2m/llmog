"""
Neo-Brutalist Theme — Gradio theme tokens (STRICT one-accent)
Pairs with console.css. One neon accent only: #FFFF00 (yellow).
Never vary accent per component; secondary stays white/black.
Background #FFFFFF, text #000000, 3px borders, 0 radius, hard shadows.
"""

import gradio as gr

# Subclassing Base via instance — Gradio's theme API is token-based, not class-based.
# We extend Base with strict brutalist tokens; CSS string in console.css
# overrides hard shadows / radius / hover transforms that the token API can't reach.

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.yellow,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
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
    # — page / surfaces: pure white, never cream/gradient —
    body_background_fill="#FFFFFF",
    body_background_fill_dark="#FFFFFF",
    body_text_color="#000000",
    body_text_color_dark="#000000",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#FFFFFF",
    background_fill_secondary="#FFFFFF",
    background_fill_secondary_dark="#FFFFFF",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    # — borders: 3px solid black everywhere (CSS enforces !important) —
    border_color_primary="#000000",
    border_color_primary_dark="#000000",
    border_color_accent="#000000",
    # — ONE accent only: #FFFF00 for primary, active tab, slider fill —
    # Secondary remains white/black, never pink/cyan/lime.
    button_primary_background_fill="#FFFF00",
    button_primary_background_fill_dark="#FFFF00",
    button_primary_background_fill_hover="#FFFF00",
    button_primary_background_fill_hover_dark="#FFFF00",
    button_primary_text_color="#000000",
    button_primary_text_color_dark="#000000",
    button_primary_border_color="#000000",
    button_primary_border_color_dark="#000000",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_background_fill_hover="#FFFFFF",
    button_secondary_background_fill_hover_dark="#FFFFFF",
    button_secondary_text_color="#000000",
    button_secondary_text_color_dark="#000000",
    button_secondary_border_color="#000000",
    button_secondary_border_color_dark="#000000",
    button_cancel_background_fill="#FFFFFF",
    button_cancel_background_fill_dark="#FFFFFF",
    button_cancel_background_fill_hover="#FFFFFF",
    button_cancel_background_fill_hover_dark="#FFFFFF",
    button_cancel_text_color="#000000",
    button_cancel_text_color_dark="#000000",
    button_cancel_border_color="#000000",
    button_cancel_border_color_dark="#000000",
    # — inputs: white, thick black border —
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#000000",
    input_border_color_dark="#000000",
    input_border_color_focus="#000000",
    input_border_color_focus_dark="#000000",
    # — slider: ONE accent yellow fill —
    slider_color="#FFFF00",
    slider_color_dark="#FFFF00",
    # — labels: black on white/yellow, never white on dark —
    block_label_text_color="#000000",
    block_label_text_color_dark="#000000",
    block_title_text_color="#000000",
    block_title_text_color_dark="#000000",
    block_label_background_fill="#FFFFFF",
    block_label_background_fill_dark="#FFFFFF",
    block_label_border_color="#000000",
    block_label_border_color_dark="#000000",
    # — radii: 0 everywhere (only tokens that exist in this Gradio version) —
    block_radius="0px",
    button_large_radius="0px",
    button_small_radius="0px",
    button_medium_radius="0px",
    input_radius="0px",
    checkbox_border_radius="0px",
    block_label_radius="0px",
    container_radius="0px",
    table_radius="0px",
    # — shadows: hard 5px/4px, never soft rgba blur —
    block_shadow="5px 5px 0px #000000",
    block_shadow_dark="5px 5px 0px #000000",
    button_primary_shadow="5px 5px 0px #000000",
    button_primary_shadow_dark="5px 5px 0px #000000",
    button_secondary_shadow="5px 5px 0px #000000",
    button_secondary_shadow_dark="5px 5px 0px #000000",
    button_cancel_shadow="5px 5px 0px #000000",
    button_cancel_shadow_dark="5px 5px 0px #000000",
    input_shadow="5px 5px 0px #000000",
    input_shadow_dark="5px 5px 0px #000000",
    input_shadow_focus="5px 5px 0px #000000",
    input_shadow_focus_dark="5px 5px 0px #000000",
    stat_background_fill="#FFFFFF",
    stat_background_fill_dark="#FFFFFF",
    table_border_color="#000000",
    table_border_color_dark="#000000",
    table_even_background_fill="#FFFFFF",
    table_even_background_fill_dark="#FFFFFF",
    table_odd_background_fill="#FFFFFF",
    table_odd_background_fill_dark="#FFFFFF",
    code_background_fill="#FFFFFF",
    code_background_fill_dark="#FFFFFF",
)
