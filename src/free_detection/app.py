"""
LLM Object Detection Console entry point wrapper.

Forwards build_app to the modular `interface` package.
"""

from interface.app_builder import build_app

__all__ = ["build_app"]

if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, inline=True)
