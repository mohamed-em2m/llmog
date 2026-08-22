"""
Retro OS Theme — Gradio theme object (Windows 95 / 98 / Classic Mac OS inspired)

Pairs with console.css. The Python theme sets Gradio's internal color variables
(tokens) to the classic 256-color desktop palette; the CSS then adds the
bevels, pixel fonts, title-bars and CRT chrome. Keep them in sync.

Palette (Win95 greybox):
  desktop        #008080  teal desktop
  window face    #c0c0c0  classic silver
  window frame   #dfdfdf / #808080 / #000000  3-D bevel
  title active   #000080  navy  (white text)
  title inactive #808080  grey  (light grey text)
  button face    #c0c0c0
  field face     #ffffff
  text           #000000
  selection      #000080  bg / #ffffff fg
"""

import gradio as gr

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    # UI font stack mirrors Win95 — MS Sans Serif → Tahoma → system sans
    font=[
        gr.themes.GoogleFont("VT323"),
        "MS Sans Serif",
        "Microsoft Sans Serif",
        "Tahoma",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("VT323"),
        gr.themes.GoogleFont("JetBrains Mono"),
        "Consolas",
        "Courier New",
        "monospace",
    ],
).set(
    # — desktop / window chrome —
    body_background_fill="#008080",
    body_background_fill_dark="#008080",
    body_text_color="#000000",
    body_text_color_dark="#000000",
    background_fill_primary="#c0c0c0",
    background_fill_primary_dark="#c0c0c0",
    background_fill_secondary="#c0c0c0",
    background_fill_secondary_dark="#c0c0c0",
    block_background_fill="#c0c0c0",
    block_background_fill_dark="#c0c0c0",
    # — borders: the CSS draws real bevels; tokens set flat fallbacks —
    border_color_primary="#000000",
    border_color_primary_dark="#000000",
    border_color_accent="#000080",
    # — buttons: classic outset silver, navy default —
    button_primary_background_fill="#c0c0c0",
    button_primary_background_fill_dark="#c0c0c0",
    button_primary_background_fill_hover="#c0c0c0",
    button_primary_background_fill_hover_dark="#c0c0c0",
    button_primary_text_color="#000000",
    button_primary_text_color_dark="#000000",
    button_primary_border_color="#000000",
    button_primary_border_color_dark="#000000",
    button_secondary_background_fill="#c0c0c0",
    button_secondary_background_fill_dark="#c0c0c0",
    button_secondary_background_fill_hover="#d4d0c8",
    button_secondary_background_fill_hover_dark="#d4d0c8",
    button_secondary_text_color="#000000",
    button_secondary_text_color_dark="#000000",
    button_secondary_border_color="#000000",
    button_secondary_border_color_dark="#000000",
    # — fields: white inset —
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#808080",
    input_border_color_dark="#808080",
    input_border_color_focus="#000080",
    input_border_color_focus_dark="#000080",
    block_label_text_color="#000000",
    block_label_text_color_dark="#000000",
    block_title_text_color="#000000",
    block_title_text_color_dark="#000000",
    block_label_background_fill="#c0c0c0",
    # — radii: retro OS has zero rounding —
    block_radius="0px",
    button_large_radius="0px",
    button_small_radius="0px",
    button_medium_radius="0px",
    input_radius="0px",
    checkbox_border_radius="0px",
    # — shadows: bevels are drawn in CSS; kill soft shadows —
    block_shadow="none",
    block_shadow_dark="none",
    button_primary_shadow="none",
    button_primary_shadow_dark="none",
    button_secondary_shadow="none",
    button_secondary_shadow_dark="none",
    input_shadow="none",
    input_shadow_dark="none",
    input_shadow_focus="none",
    input_shadow_focus_dark="none",
)
