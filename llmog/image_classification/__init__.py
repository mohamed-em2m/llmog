"""Whole-image classification (CLI task ``classify``).

Send images in, get a CSV out::

    llmog --task classify -i img1.jpg --input_folder ./imgs \\
        -c "hole, stain, good" --output_format csv -o ./classify_results

Public surface mirrors the other tasks:

* :func:`main` -- accepts a validated ``PipelineConfig``, a legacy argparse
  ``Namespace``, or ``None`` (parse ``sys.argv`` via :func:`build_parser`).
* :func:`build_parser` -- standalone argparse parser for ``classify-cli``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("llmog.classify")

__all__ = ["main", "build_parser"]


# ---------------------------------------------------------------------------
# Standalone parser (mirrors the classify subset of PipelineConfig)
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="classify-cli",
        description=(
            "Classify whole images with a VLM and write predictions.csv "
            "(optionally YOLO-cls .txt files)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image",
        "-i",
        metavar="PATH",
        action="append",
        dest="images",
        default=None,
        help="Input image path. Repeatable.",
    )
    p.add_argument(
        "--input_folder", default=None, help="Folder of images (combined with --image)."
    )
    p.add_argument("--image_extensions", default=".jpg,.jpeg,.png")
    p.add_argument("--categories", "-c", default="person, car, bicycle, dog, cat")
    p.add_argument("--definitions", "-d", default="")
    p.add_argument(
        "--class_mode", choices=["strict", "hybrid", "free"], default="strict"
    )
    p.add_argument("--class_definitions", default="")
    p.add_argument(
        "--preset",
        default=None,
        choices=[
            "fabric_defects",
            "coco",
            "road_traffic",
            "retail_packaging",
            "pcb_defects",
        ],
    )
    p.add_argument(
        "--classification_mode", choices=["single", "multi", "top_k"], default="single"
    )
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--multi_threshold", type=float, default=50.0)
    p.add_argument("--output_format", choices=["csv", "yolo", "both"], default="csv")
    p.add_argument("--output_folder", "-o", default="./classify_results")
    p.add_argument("--model", default="local-model")
    p.add_argument("--api_key", default="not-needed")
    p.add_argument("--base_url", default="http://localhost:8080/v1")
    p.add_argument(
        "--server_type",
        default="llama_cpp",
        choices=["llama_cpp", "llama_cpp_python", "vllm", "external"],
    )
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--ctx_size", type=int, default=20000)
    p.add_argument("--parallel_slots", type=int, default=1)
    p.add_argument("--max_workers", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start_index", type=int, default=None)
    p.add_argument("--end_index", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--auto_resume", action="store_true", default=True)
    p.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    p.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    p.add_argument("--log_file", default=None)
    p.add_argument("--classification_temperature", type=float, default=0.2)
    p.add_argument("--classification_max_tokens", type=int, default=1024)
    p.add_argument("--api_retries", type=int, default=3)
    return p


def _as_dict(args: Any) -> Dict[str, Any]:
    if hasattr(args, "model_dump"):
        return dict(args.model_dump())
    if isinstance(args, dict):
        return dict(args)
    return dict(vars(args))


def _resolve_preset_defs(preset: str | None, class_definitions: str) -> str:
    from image_classification.prompt import resolve_class_definitions

    return resolve_class_definitions(class_definitions, preset)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main(args: Any = None) -> None:
    """Run whole-image classification.

    ``args`` may be a ``PipelineConfig``, an argparse ``Namespace`` from
    :func:`build_parser`, or ``None`` (parse ``sys.argv`` standalone).
    """
    if args is None:
        ns = build_parser().parse_args()
        raw = {k: v for k, v in vars(ns).items() if v is not None}
        raw["task"] = "classify"
        try:
            from schemes import PipelineConfig

            args = PipelineConfig(**raw)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(2)

    cfg = _as_dict(args)
    logging.basicConfig(
        level=getattr(logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO),
        format="[%(levelname)s] %(message)s",
    )

    from image_classification.inputs import collect_classify_images
    from image_classification.prompt import (
        parse_categories,
        render_image_classify_prompt,
    )
    from image_classification.inference import classify_single_image
    from image_classification.outputs import (
        build_class_index,
        write_classes_txt,
        write_predictions_csv,
        write_summary_json,
        write_yolo_cls,
    )

    # --- Categories / definitions -----------------------------------------
    categories = parse_categories(str(cfg.get("categories") or ""))
    preset = cfg.get("preset")
    class_definitions = _resolve_preset_defs(
        preset, str(cfg.get("class_definitions") or cfg.get("definitions") or "")
    )
    class_mode = str(cfg.get("class_mode") or "strict")
    if class_mode == "strict" and not categories:
        print("ERROR: strict class_mode requires --categories.", file=sys.stderr)
        sys.exit(1)

    classification_mode = str(cfg.get("classification_mode") or "single")
    top_k = int(cfg.get("top_k") or 3)
    multi_threshold = float(cfg.get("multi_threshold", 50.0))
    output_format = str(cfg.get("output_format") or "csv")
    out_base = Path(str(cfg.get("output_folder") or "./classify_results"))
    out_base.mkdir(parents=True, exist_ok=True)
    dry_run = bool(cfg.get("dry_run", False))
    max_workers = max(1, int(cfg.get("max_workers") or 1))

    # --- Collect inputs -----------------------------------------------------
    targets = collect_classify_images(
        images=cfg.get("images"),
        input_folder=cfg.get("input_folder"),
        image_extensions=str(cfg.get("image_extensions") or ".jpg,.jpeg,.png"),
        shuffle=bool(cfg.get("shuffle", False)),
        seed=int(cfg.get("seed") or 42),
        start_index=cfg.get("start_index"),
        end_index=cfg.get("end_index"),
        num_samples=cfg.get("num_samples"),
    )
    # Warn about missing explicit -i paths (collect skips them silently)
    for raw in cfg.get("images") or []:
        if not Path(raw).is_file():
            print(f"WARNING: image not found, skipping: {raw}", file=sys.stderr)
    if not targets:
        print(
            "No images to classify (check --image / --input_folder).", file=sys.stderr
        )
        sys.exit(1)

    # --- Checkpoint / resume ------------------------------------------------
    try:
        from auto_annotation.checkpoint import CheckpointManager

        checkpoint = CheckpointManager(str(out_base))
    except Exception:
        checkpoint = None  # type: ignore[assignment]
    auto_resume = bool(cfg.get("auto_resume", True))
    completed: set[str] = set()
    if checkpoint is not None:
        if not auto_resume:
            checkpoint.clear()
        else:
            data = checkpoint.load()
            if data:
                completed = set(data.get("completed_images", []))
                print(f"Auto-resume: {len(completed)} image(s) already done.")

    pending = [p for p in targets if p.stem not in completed]
    skipped_resume = len(targets) - len(pending)
    if skipped_resume:
        print(f"Skipping {skipped_resume} image(s) per checkpoint.")

    prompt = render_image_classify_prompt(
        categories=categories,
        class_mode=class_mode,
        class_definitions=class_definitions,
        classification_mode=classification_mode,
        top_k=top_k,
        multi_threshold=multi_threshold,
    )

    if dry_run:
        print(f"[dry run] would classify {len(pending)} image(s) -> {out_base}")
        for p in pending:
            print(f"  {p}")
        return

    # --- Model client -------------------------------------------------------
    try:
        from auto_annotation.server_init import build_client
    except Exception as exc:
        print(f"ERROR: could not load server backend: {exc}", file=sys.stderr)
        sys.exit(1)

    model_name = str(cfg.get("model") or "local-model")
    try:
        client, manager = build_client(
            args if not isinstance(args, dict) else _ns_from_cfg(cfg)
        )
    except Exception as exc:
        print(f"ERROR: failed to initialize server/client: {exc}", file=sys.stderr)
        sys.exit(1)

    temperature = float(cfg.get("classification_temperature", 0.2))
    max_tokens = int(cfg.get("classification_max_tokens", 1024))
    retries = int(cfg.get("api_retries", 3))
    lock = threading.Lock()
    rows: List[Dict[str, Any]] = []
    all_preds: List[List[Dict[str, Any]]] = []

    def _run_one(image_path: Path) -> Dict[str, Any]:
        try:
            preds = classify_single_image(
                image_path,
                client,
                model_name,
                prompt,
                classification_mode=classification_mode,
                top_k=top_k,
                multi_threshold=multi_threshold,
                temperature=temperature,
                max_tokens=max_tokens,
                retries=retries,
            )
            best = preds[0]
            return {
                "path": image_path,
                "status": "ok",
                "preds": preds,
                "row": {
                    "filename": image_path.name,
                    "path": str(image_path),
                    "predicted_class": best["class"],
                    "confidence": f"{best['confidence']:.1f}",
                    "reasoning": best["reasoning"],
                    "top_k_json": json.dumps(preds),
                    "all_scores_json": json.dumps(preds),
                    "status": "ok",
                    "error": "",
                },
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed %s: %s", image_path, exc)
            return {
                "path": image_path,
                "status": "error",
                "preds": [],
                "row": {
                    "filename": image_path.name,
                    "path": str(image_path),
                    "predicted_class": "error",
                    "confidence": "0.0",
                    "reasoning": "",
                    "top_k_json": "[]",
                    "all_scores_json": "[]",
                    "status": "error",
                    "error": str(exc)[:300],
                },
            }

    try:
        if max_workers <= 1:
            ordered = [_run_one(p) for p in pending]
        else:
            ordered_map: Dict[str, Dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_run_one, p): p for p in pending}
                for fut in as_completed(futs):
                    res = fut.result()
                    ordered_map[str(res["path"])] = res
            ordered = [ordered_map[str(p)] for p in pending]

        for res in ordered:
            rows.append(res["row"])
            all_preds.append(res["preds"])
            if checkpoint is not None and res["status"] == "ok":
                with lock:
                    completed.add(Path(str(res["row"]["path"])).stem)
                    try:
                        checkpoint.save(set(completed), {}, set())
                    except Exception:
                        pass
            mark = "OK" if res["status"] == "ok" else "ERR"
            print(
                f"  [{mark}] {res['row']['filename']} -> "
                f"{res['row']['predicted_class']} ({res['row']['confidence']})"
            )
    finally:
        try:
            if manager is not None:
                manager.stop_llama_server()
        except Exception:
            try:
                if manager is not None and hasattr(manager, "stop_vllm_server"):
                    manager.stop_vllm_server()  # type: ignore[attr-defined]
            except Exception:
                pass

    # --- Outputs --------------------------------------------------------------
    class_index = build_class_index(categories, all_preds)
    wrote: List[str] = []
    if output_format in ("csv", "both"):
        csv_path = write_predictions_csv(rows, out_base / "predictions.csv")
        wrote.append(str(csv_path))
    if output_format in ("yolo", "both"):
        labels_dir = out_base / "labels"
        write_yolo_cls(rows, class_index, labels_dir)
        write_classes_txt(class_index, out_base / "classes.txt")
        wrote.append(str(labels_dir))
    write_summary_json(rows, class_index, out_base / "summary.json")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"\nDone: {ok}/{len(rows)} ok. Wrote: {', '.join(wrote)}")


def _ns_from_cfg(cfg: Dict[str, Any]) -> argparse.Namespace:
    """Adapt a plain dict config to the Namespace API build_client expects."""
    return argparse.Namespace(**cfg)


if __name__ == "__main__":
    main()
