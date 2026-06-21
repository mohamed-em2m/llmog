"""
Gradio API Console — theme object.

Paste this into your app (or `from assets.console_theme import theme` if you
keep the file alongside your app code) and pass it to gr.Blocks(theme=theme).

This is the gr.themes.Base() override that gives the dark GitHub-style look.
It's deliberately separate from console.css: this theme object controls
Gradio's *internal* component defaults (the ones CSS alone can't always
reach, like dark-mode variants and some input chrome), while console.css
handles layout, typography, and the custom output-panel/JSON-panel/history
structures. Load both — the theme alone looks decent but flat; the CSS
alone fights Gradio's light-mode defaults in places. Together they match.
"""

import gradio as gr

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.sky,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont('JetBrains Mono'), 'monospace'],
    font_mono=[gr.themes.GoogleFont('JetBrains Mono'), 'monospace'],
).set(
    body_background_fill='#080d14',
    body_background_fill_dark='#080d14',
    body_text_color='#c9d1d9',
    body_text_color_dark='#c9d1d9',
    background_fill_primary='#0d1117',
    background_fill_primary_dark='#0d1117',
    background_fill_secondary='#161b22',
    background_fill_secondary_dark='#161b22',
    border_color_primary='#21262d',
    border_color_primary_dark='#21262d',
    button_primary_background_fill='#0ea5e9',
    button_primary_background_fill_hover='#38bdf8',
    button_primary_text_color='#0a0f1a',
    button_secondary_background_fill='#161b22',
    button_secondary_background_fill_hover='#21262d',
    button_secondary_text_color='#c9d1d9',
    button_secondary_border_color='#30363d',
    input_background_fill='#0d1117',
    input_background_fill_dark='#0d1117',
    input_border_color='#21262d',
    input_border_color_focus='#38bdf8',
    block_background_fill='#0d1117',
    block_background_fill_dark='#0d1117',
    block_border_color='#21262d',
    block_label_text_color='#7d8590',
    block_title_text_color='#38bdf8',
)
