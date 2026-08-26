"""
Bauhaus Guitar — Gradio theme tokens (Bauhaus-inspired guitar learning app)
Pairs with console.css. Warm cream base, bold geometric accents.

Color system (from prompt):
  Core:       --color-bg            #FFF8EC (warm cream page)
              --color-surface       #FFFFFF (cards/modals/inputs)
              --color-text-primary  #1A1A1A (headings/body)
              --color-text-secondary #6B6B6B (captions/metadata)
  Accent:     --color-accent-pink   #E84B8A (guitar illustration/highlights)
              --color-accent-yellow #F5C842 (CTA/buttons/badges/active)
              --color-accent-orange #E8622A (icons/buttons/progress/rings)
              --color-accent-navy   #1A1145 (dark lesson cards)
              --color-accent-teal   #1B7B6E (corner triangles)
              --color-accent-blue   #2D4EAA (background shapes)
  UI:         --color-btn-dark      #1E1E1E (pill button bg)
              --color-btn-text      #F5C842 (yellow on dark)
              --color-tab-active    #FFD6E0 (soft pink active pill)
              --color-progress-bar  #E8622A (orange)
              --color-ring-outer    #E8622A
              --color-ring-middle   #F5C842
              --color-ring-inner    #1A1145
Rounded: cards 16-20px, pills 999px, soft shadows — NOT brutal 3px/0-radius/hard.
"""

import gradio as gr

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,  # closest to #F5C842 yellow
    secondary_hue=gr.themes.colors.blue,  # #2D4EAA blue
    neutral_hue=gr.themes.colors.stone,  # warm cream neutral
    font=[
        gr.themes.GoogleFont("Inter"),
        gr.themes.GoogleFont("DM Sans"),
        gr.themes.GoogleFont("Poppins"),
        "Inter",
        "DM Sans",
        "Poppins",
        "Space Grotesk",
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
    # — page / surfaces — warm cream —
    body_background_fill="#FFF8EC",
    body_background_fill_dark="#FFF8EC",
    body_text_color="#1A1A1A",
    body_text_color_dark="#1A1A1A",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#FFFFFF",
    background_fill_secondary="#FFF8EC",
    background_fill_secondary_dark="#FFF8EC",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    # — borders — soft warm stone, 1px —
    border_color_primary="#E9E0CC",
    border_color_primary_dark="#E9E0CC",
    border_color_accent="#E84B8A",
    # — buttons — dark pill + yellow text (Bauhaus DNA) —
    button_primary_background_fill="#1E1E1E",
    button_primary_background_fill_dark="#1E1E1E",
    button_primary_background_fill_hover="#2A2A2A",
    button_primary_background_fill_hover_dark="#2A2A2A",
    button_primary_text_color="#F5C842",
    button_primary_text_color_dark="#F5C842",
    button_primary_border_color="#1E1E1E",
    button_primary_border_color_dark="#1E1E1E",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_background_fill_hover="#FFF8EC",
    button_secondary_background_fill_hover_dark="#FFF8EC",
    button_secondary_text_color="#1A1A1A",
    button_secondary_text_color_dark="#1A1A1A",
    button_secondary_border_color="#E9E0CC",
    button_secondary_border_color_dark="#E9E0CC",
    button_cancel_background_fill="#FFD6E0",
    button_cancel_background_fill_dark="#FFD6E0",
    button_cancel_background_fill_hover="#FFE4EC",
    button_cancel_background_fill_hover_dark="#FFE4EC",
    button_cancel_text_color="#1A1A1A",
    button_cancel_text_color_dark="#1A1A1A",
    button_cancel_border_color="#E9E0CC",
    button_cancel_border_color_dark="#E9E0CC",
    # — inputs — white rounded 12px —
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#E9E0CC",
    input_border_color_dark="#E9E0CC",
    input_border_color_focus="#F5C842",
    input_border_color_focus_dark="#F5C842",
    # — slider — orange progress / yellow accent —
    slider_color="#E8622A",
    slider_color_dark="#E8622A",
    # — labels — warm secondary —
    block_label_text_color="#6B6B6B",
    block_label_text_color_dark="#6B6B6B",
    block_title_text_color="#1A1A1A",
    block_title_text_color_dark="#1A1A1A",
    block_label_background_fill="#FFF8EC",
    block_label_background_fill_dark="#FFF8EC",
    block_label_border_color="#E9E0CC",
    block_label_border_color_dark="#E9E0CC",
    # — radii — Bauhaus rounded —
    block_radius="16px",
    button_large_radius="999px",
    button_small_radius="999px",
    button_medium_radius="999px",
    input_radius="12px",
    checkbox_border_radius="6px",
    block_label_radius="8px",
    container_radius="20px",
    table_radius="12px",
    # — shadows — soft layered —
    block_shadow="0 4px 16px rgba(26,17,69,0.08)",
    block_shadow_dark="0 4px 16px rgba(26,17,69,0.08)",
    button_primary_shadow="0 4px 12px rgba(26,17,69,0.12)",
    button_primary_shadow_dark="0 4px 12px rgba(26,17,69,0.12)",
    button_secondary_shadow="0 1px 4px rgba(26,17,69,0.06)",
    button_secondary_shadow_dark="0 1px 4px rgba(26,17,69,0.06)",
    button_cancel_shadow="0 1px 4px rgba(26,17,69,0.06)",
    button_cancel_shadow_dark="0 1px 4px rgba(26,17,69,0.06)",
    input_shadow="0 1px 2px rgba(26,17,69,0.04)",
    input_shadow_dark="0 1px 2px rgba(26,17,69,0.04)",
    input_shadow_focus="0 0 0 3px rgba(245,200,66,0.30)",
    input_shadow_focus_dark="0 0 0 3px rgba(245,200,66,0.30)",
    stat_background_fill="#FFFFFF",
    stat_background_fill_dark="#FFFFFF",
    table_border_color="#E9E0CC",
    table_border_color_dark="#E9E0CC",
    table_even_background_fill="#FFFFFF",
    table_even_background_fill_dark="#FFFFFF",
    table_odd_background_fill="#FFF8EC",
    table_odd_background_fill_dark="#FFF8EC",
    code_background_fill="#FFF8EC",
    code_background_fill_dark="#FFF8EC",
)
