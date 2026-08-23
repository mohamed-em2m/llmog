"""
Retro-Playful Guitar Theme — Gradio theme tokens
Palette inspired by guitar learning app:
  cream  #FFF8EC  main bg
  charcoal #1E1E1E text / dark buttons / card bg
  magenta #E84B8A  accent (guitar, active tab)
  yellow  #F5C842  buttons/badges/interactive
  orange  #D4641A  icons inside buttons
  navy    #1A1145  "Continue" cards
  soft-pink #F9C6D2 active nav pill
  light-gray #A0A0A0 inactive icons
Pairs with console.css which does the heavy lifting (pill shapes, soft corners).
"""

import gradio as gr

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.yellow,
    secondary_hue=gr.themes.colors.pink,
    neutral_hue=gr.themes.colors.stone,
    font=[
        gr.themes.GoogleFont("Space Grotesk"),
        gr.themes.GoogleFont("Outfit"),
        "Space Grotesk",
        "Outfit",
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
    # — page — warm cream, cozy retro
    body_background_fill="#FFF8EC",
    body_background_fill_dark="#FFF8EC",
    body_text_color="#1E1E1E",
    body_text_color_dark="#1E1E1E",
    background_fill_primary="#FFF8EC",
    background_fill_primary_dark="#FFF8EC",
    background_fill_secondary="#FFFFFF",
    background_fill_secondary_dark="#FFFFFF",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    # — borders — charcoal, not pure black
    border_color_primary="#1E1E1E",
    border_color_primary_dark="#1E1E1E",
    border_color_accent="#E84B8A",
    # — buttons: dark pill (charcoal bg, yellow text) vs yellow pill —
    button_primary_background_fill="#1E1E1E",
    button_primary_background_fill_dark="#1E1E1E",
    button_primary_background_fill_hover="#2A2A2A",
    button_primary_background_fill_hover_dark="#2A2A2A",
    button_primary_text_color="#F5C842",
    button_primary_text_color_dark="#F5C842",
    button_primary_border_color="#1E1E1E",
    button_primary_border_color_dark="#1E1E1E",
    button_secondary_background_fill="#F5C842",
    button_secondary_background_fill_dark="#F5C842",
    button_secondary_background_fill_hover="#FFD65A",
    button_secondary_background_fill_hover_dark="#FFD65A",
    button_secondary_text_color="#1E1E1E",
    button_secondary_text_color_dark="#1E1E1E",
    button_secondary_border_color="#1E1E1E",
    button_secondary_border_color_dark="#1E1E1E",
    button_cancel_background_fill="#FFFFFF",
    button_cancel_background_fill_dark="#FFFFFF",
    button_cancel_background_fill_hover="#F9C6D2",
    button_cancel_background_fill_hover_dark="#F9C6D2",
    button_cancel_text_color="#1E1E1E",
    button_cancel_text_color_dark="#1E1E1E",
    button_cancel_border_color="#1E1E1E",
    button_cancel_border_color_dark="#1E1E1E",
    # — inputs — white on cream, charcoal border
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#1E1E1E",
    input_border_color_dark="#1E1E1E",
    input_border_color_focus="#E84B8A",
    input_border_color_focus_dark="#E84B8A",
    # — slider — yellow progress on navy/charcoal
    slider_color="#F5C842",
    slider_color_dark="#F5C842",
    # — labels — charcoal on cream, yellow on dark
    block_label_text_color="#1E1E1E",
    block_label_text_color_dark="#1E1E1E",
    block_title_text_color="#1E1E1E",
    block_title_text_color_dark="#1E1E1E",
    block_label_background_fill="#F9C6D2",
    block_label_background_fill_dark="#F9C6D2",
    block_label_border_color="#1E1E1E",
    block_label_border_color_dark="#1E1E1E",
    # — radii — pill + soft corners (retro playful, not brutal 0)
    block_radius="20px",
    button_large_radius="999px",
    button_small_radius="999px",
    button_medium_radius="999px",
    input_radius="14px",
    checkbox_border_radius="10px",
    block_label_radius="999px",
    container_radius="24px",
    table_radius="16px",
    # — shadows — soft warm, not hard 5px brutal
    block_shadow="0 6px 0 #1E1E1E",
    block_shadow_dark="0 6px 0 #1E1E1E",
    button_primary_shadow="0 4px 0 #1E1E1E",
    button_primary_shadow_dark="0 4px 0 #1E1E1E",
    button_secondary_shadow="0 4px 0 #1E1E1E",
    button_secondary_shadow_dark="0 4px 0 #1E1E1E",
    button_cancel_shadow="0 4px 0 #1E1E1E",
    button_cancel_shadow_dark="0 4px 0 #1E1E1E",
    input_shadow="0 3px 0 #1E1E1E",
    input_shadow_dark="0 3px 0 #1E1E1E",
    input_shadow_focus="0 4px 0 #E84B8A",
    input_shadow_focus_dark="0 4px 0 #E84B8A",
    stat_background_fill="#FFFFFF",
    stat_background_fill_dark="#FFFFFF",
    table_border_color="#1E1E1E",
    table_border_color_dark="#1E1E1E",
    table_even_background_fill="#FFF8EC",
    table_even_background_fill_dark="#FFF8EC",
    table_odd_background_fill="#FFFFFF",
    table_odd_background_fill_dark="#FFFFFF",
    code_background_fill="#FFF8EC",
    code_background_fill_dark="#FFF8EC",
)
