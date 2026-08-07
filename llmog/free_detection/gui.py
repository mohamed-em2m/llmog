"""
GUI entry point — launches the Gradio web interface.

Usage:
    uv run detection-gui
    uv run detection-gui --share          # public Gradio link
    uv run detection-gui --port 7861      # custom port
"""

from __future__ import annotations

import argparse
from .app import build_app


def main():
    parser = argparse.ArgumentParser(
        prog="detection-gui",
        description="Launch the Gradio web UI for the LLM Object Detection tester.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the Gradio server to."
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to run the Gradio server on."
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link."
    )
    parser.add_argument(
        "--no-queue", action="store_true", help="Disable Gradio's request queue."
    )
    args = parser.parse_args()

    # imported here so --help is instant
    demo = build_app()
    if args.no_queue:
        demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    else:
        demo.queue().launch(
            server_name=args.host, server_port=args.port, share=args.share
        )


if __name__ == "__main__":
    main()
