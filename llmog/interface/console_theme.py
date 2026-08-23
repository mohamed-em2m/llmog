"""
Neo-Brutalism Theme — Gradio theme object
Pairs with console.css (brutalist hard-shadow, thick borders, acid accents).

Palette:
  ink          #0a0a0a  text / borders / shadows
  paper        #ffffff  card face
  cream        #FFFDF0  page bg
  yellow       #FFE600  primary (CTA, selected tab)
  pink         #FF90E8  hover / secondary pop
  cyan         #2EEBFF  info
  lime         #C8FF00  success
  orange       #FF6B2D  warning
  red          #FF3B30  danger
"""

import gradio as gr

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.yellow,
    secondary_hue=gr.themes.colors.pink,
    neutral_hue=gr.themes.colors.stone,
    font=[
        gr.themes.GoogleFont("Space Grotesk"),
        "Space Grotesk",
        "Inter",
        "system-ui",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("Space Mono"),
        gr.themes.GoogleFont("JetBrains Mono"),
        "Space Mono",
        "Consolas",
        "monospace",
    ],
).set(
    # — page / cards —
    body_background_fill="#FFFDF0",
    body_background_fill_dark="#FFFDF0",
    body_text_color="#0a0a0a",
    body_text_color_dark="#0a0a0a",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#FFFDF0",
    background_fill_secondary_dark="#FFFDF0",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    # — borders: thick brutal 3px —
    border_color_primary="#0a0a0a",
    border_color_primary_dark="#0a0a0a",
    border_color_accent="#0a0a0a",
    # — buttons: yellow primary, white secondary —
    button_primary_background_fill="#FFE600",
    button_primary_background_fill_dark="#FFE600",
    button_primary_background_fill_hover="#FF90E8",
    button_primary_background_fill_hover_dark="#FF90E8",
    button_primary_text_color="#0a0a0a",
    button_primary_text_color_dark="#0a0a0a",
    button_primary_border_color="#0a0a0a",
    button_primary_border_color_dark="#0a0a0a",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#ffffff",
    button_secondary_background_fill_hover="#FF90E8",
    button_secondary_background_fill_hover_dark="#FF90E8",
    button_secondary_text_color="#0a0a0a",
    button_secondary_text_color_dark="#0a0a0a",
    button_secondary_border_color="#0a0a0a",
    button_secondary_border_color_dark="#0a0a0a",
    # — fields: white with thick border —
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#0a0a0a",
    input_border_color_dark="#0a0a0a",
    input_border_color_focus="#0a0a0a",
    input_border_color_focus_dark="#0a0a0a",
    block_label_text_color="#0a0a0a",
    block_label_text_color_dark="#0a0a0a",
    block_title_text_color="#0a0a0a",
    block_title_text_color_dark="#0a0a0a",
    block_label_background_fill="#0a0a0a",
    # — radii: brutal has zero rounding —
    block_radius="0px",
    button_large_radius="0px",
    button_small_radius="0px",
    button_medium_radius="0px",
    input_radius="0px",
    checkbox_border_radius="0px",
    # — shadows: hard offset, no blur —
    block_shadow="5px 5px 0 #0a0a0a",
    block_shadow_dark="5px 5px 0 #0a0a0a",
    button_primary_shadow="4px 4px 0 #0a0a0a",
    button_primary_shadow_dark="4px 4px 0 #0a0a0a",
    button_secondary_shadow="4px 4px 0 #0a0a0a",
    button_secondary_shadow_dark="4px 4px 0 #0a0a0a",
    input_shadow="3px 3px 0 #0a0a0a",
    input_shadow_dark="3px 3px 0 #0a0a0a",
    input_shadow_focus="4px 4px 0 #0a0a0a",
    input_shadow_focus_dark="4px 4px 0 #0a0a0a",
)
