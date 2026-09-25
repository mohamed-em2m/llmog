"""Reverse batched auto-label output back into a flat, resumable state.

Batched runs write ``<output>/labels/batch_XXXX/*.txt``. If the run was
stopped with ``--no_auto_resume`` there is no ``.checkpoint.json``, so a
plain re-run would redo everything and reshuffle class ids.

This module reverses that situation:

1. Scans all ``batch_*/`` label files, groups by stem
   (txt stem == image stem, e.g. ``002674_jpg.rf.<hash>``).
2. Picks the best copy per stem: non-empty > newest mtime > largest size.
3. Writes a flat ``labels_flat/`` folder (YOLO-trainable layout).
4. Rebuilds ``.checkpoint.json`` (``completed_images`` + ``class_map``
   from ``data.yaml``, ``batches_done=[]``) so the next run skips finished
   images per-image instead of redoing them.
5. Optionally drops junk classes (e.g. ``--drop-class-names ...``) and
   compacts the remaining ids, rewriting labels + ``data.yaml``.

Usage:
    python -m auto_annotation.reverse_batches --output_folder ./output
    python -m auto_annotation.reverse_batches --output_folder ./output \\
        --drop-class-names ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from auto_annotation.logging_utils import logger, setup_logging


def pick_best(paths: list[Path]) -> Path:
    """Best copy per stem: non-empty > newest mtime > largest size.

    Deterministic tiebreak on parent name so repeated runs agree.
    """

    def _score(p: Path):
        try:
            st = p.stat()
            txt = p.read_text(encoding="utf-8").strip()
        except Exception:
            return (0, 0, 0, "")
        return (1 if txt else 0, st.st_mtime, st.st_size, p.parent.name)

    return max(paths, key=_score)


def group_by_stem(labels_dir: Path) -> dict[str, list[Path]]:
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for p in labels_dir.rglob("*.txt"):
        if p.parent.name == labels_dir.name and labels_dir.parent is not None:
            # flat file directly under labels/ (batch_size=0 runs) — keep too
            pass
        by_stem[p.stem].append(p)
    return by_stem


def load_class_map(data_yaml: Path) -> tuple[dict, list]:
    data = yaml.safe_load(open(data_yaml, encoding="utf-8")) or {}
    names = data.get("names", []) or []
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    class_map = {str(n): i for i, n in enumerate(names)}
    return class_map, list(names), data


def rebuild_checkpoint(
    output_folder: str | Path,
    labels_dirname: str = "labels",
    data_yaml_name: str = "data.yaml",
    checkpoint_name: str = ".checkpoint.json",
    dry_run: bool = False,
) -> dict:
    """Rebuild checkpoint from existing batch files. Returns the payload."""
    output = Path(output_folder)
    labels_dir = output / labels_dirname
    data_yaml = output / data_yaml_name
    ckpt = output / checkpoint_name
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels dir missing: {labels_dir}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml missing: {data_yaml}")

    class_map, names, _ = load_class_map(data_yaml)
    logger.info(f"class_map from {data_yaml} ({len(class_map)}): {class_map}")

    by_stem = group_by_stem(labels_dir)
    total = sum(len(v) for v in by_stem.values())
    dups = sum(1 for v in by_stem.values() if len(v) > 1)
    logger.info(f"total .txt: {total}, unique stems: {len(by_stem)}, dups: {dups}")

    payload = {
        "completed_images": sorted(by_stem.keys()),
        "class_map": dict(class_map),
        "batches_done": [],
    }
    if dry_run:
        logger.info(f"[dry run] would write {ckpt} ({len(by_stem)} completed).")
        return payload
    tmp = ckpt.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, ckpt)  # atomic like CheckpointManager.save
    logger.info(f"Wrote {ckpt} ({len(by_stem)} completed, batches_done=[]).")
    return payload


def consolidate_batches(
    output_folder: str | Path,
    labels_dirname: str = "labels",
    flat_dirname: str = "labels_flat",
    copy_low_conf: bool = True,
    dry_run: bool = False,
) -> Path:
    """Copy best copy per stem into a flat folder. Returns flat dir path."""
    output = Path(output_folder)
    labels_dir = output / labels_dirname
    flat = output / flat_dirname
    by_stem = group_by_stem(labels_dir)
    if dry_run:
        logger.info(f"[dry run] would write {len(by_stem)} files to {flat}.")
        return flat
    flat.mkdir(parents=True, exist_ok=True)
    for stem, paths in by_stem.items():
        best = pick_best(paths)
        shutil.copy2(best, flat / (stem + ".txt"))
    logger.info(f"Wrote flat {len(by_stem)} files -> {flat}.")
    if copy_low_conf:
        n = 0
        for p in labels_dir.rglob("*_low_confidence.json"):
            dest = flat / f"{p.parent.name}_{p.name}"
            if not dest.exists():
                shutil.copy2(p, dest)
                n += 1
        logger.info(f"Copied {n} low_confidence json(s).")
    return flat


def flatten_batches_to_labels(
    output_folder: str | Path,
    labels_dirname: str = "labels",
    copy_low_conf: bool = True,
    dry_run: bool = False,
) -> Path:
    """Copy best copy per stem to the TOP LEVEL of the labels dir.

    Batched runs write ``<output>/labels/batch_XXXX/*.txt`` which is not
    YOLO-trainable (YOLO expects flat ``*.txt`` next to the images). After
    all batches finish, this copies the best copy per stem
    (non-empty > newest > largest, see :func:`pick_best`) to
    ``<output>/labels/<stem>.txt`` so the labels dir is directly usable for
    training. Copy-only: ``batch_XXXX/`` dirs and ``.checkpoint.json`` stay
    untouched, so resume (``batches_done`` skipping) keeps working and a
    re-run is idempotent (already-flattened stems are skipped).
    """
    output = Path(output_folder)
    labels_dir = output / labels_dirname
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"labels dir missing: {labels_dir}")
    by_stem = group_by_stem(labels_dir)
    if dry_run:
        logger.info(f"[dry run] would flatten {len(by_stem)} files -> {labels_dir}.")
        return labels_dir
    n_copied = 0
    for stem, paths in by_stem.items():
        best = pick_best(paths)
        dest = labels_dir / (stem + ".txt")
        try:
            if best.resolve() == dest.resolve():
                continue  # already flat (batch_size=0 runs / previous flatten)
        except Exception:
            if best == dest:
                continue
        shutil.copy2(best, dest)
        n_copied += 1
    logger.info(
        f"Flattened {n_copied} file(s) (best copy per stem, "
        f"{len(by_stem)} total) -> {labels_dir}."
    )
    if copy_low_conf:
        n = 0
        for p in labels_dir.rglob("*_low_confidence.json"):
            if p.parent == labels_dir:
                continue  # already top-level
            dest = labels_dir / f"{p.parent.name}_{p.name}"
            if not dest.exists():
                shutil.copy2(p, dest)
                n += 1
        logger.info(f"Copied {n} low_confidence json(s) -> {labels_dir}.")
    return labels_dir


def drop_classes_and_compact(
    labels_folder: str | Path,
    data_yaml: str | Path,
    drop_names: list[str],
    dry_run: bool = False,
) -> dict:
    """Drop boxes whose class name is in drop_names, compact ids, rewrite.

    Returns the new ``{name: id}`` map. Operates on a FLAT labels folder
    (use :func:`consolidate_batches` output) plus its ``data.yaml`` so the
    original ``batch_*/`` output is never mutated.
    """
    labels_folder = Path(labels_folder)
    data_yaml = Path(data_yaml)
    class_map, names, data = load_class_map(data_yaml)
    drop = {str(x).strip() for x in drop_names if str(x).strip()}
    keep = [n for n in names if n not in drop]
    if len(keep) == len(names):
        logger.info(f"Nothing to drop (drop={sorted(drop)} not in {names}).")
        return class_map
    old_id_of = {n: i for i, n in enumerate(names)}
    new_id_of = {n: i for i, n in enumerate(keep)}
    remap = {old_id_of[n]: new_id_of[n] for n in keep}
    logger.info(f"Dropping {sorted(drop)}: {names} -> {keep}")

    txts = list(labels_folder.glob("*.txt"))
    if dry_run:
        logger.info(f"[dry run] would rewrite {len(txts)} files + {data_yaml}.")
        return new_id_of
    for p in txts:
        lines = p.read_text(encoding="utf-8").splitlines()
        kept = []
        for ln in lines:
            if not ln.strip():
                continue
            parts = ln.split()
            try:
                cid = int(float(parts[0]))
            except ValueError:
                continue
            if cid in remap:
                kept.append(f"{remap[cid]} {' '.join(parts[1:])}")
        p.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    updated = dict(data)
    updated["names"] = keep
    updated["nc"] = len(keep)
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(updated, f, sort_keys=False)
    logger.info(f"Rewrote {len(txts)} labels + {data_yaml} (nc={len(keep)}).")
    return new_id_of


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Reverse batch_XXXX auto-label output: flat labels + checkpoint rebuild."
    )
    ap.add_argument("--output_folder", required=True, help="Run output folder.")
    ap.add_argument("--labels_dirname", default="labels")
    ap.add_argument("--flat_dirname", default="labels_flat")
    ap.add_argument("--data_yaml_name", default="data.yaml")
    ap.add_argument(
        "--no_checkpoint", action="store_true", help="Skip .checkpoint.json rebuild."
    )
    ap.add_argument(
        "--no_flat", action="store_true", help="Skip flat labels consolidation."
    )
    ap.add_argument(
        "--drop-class-names",
        dest="drop_class_names",
        default="",
        help="Comma-separated class NAMES to drop+compact (e.g. '...'). "
        "Applies to the flat folder + a copied data.yaml next to it.",
    )
    ap.add_argument("--dry_run", action="store_true")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_logging(log_level="INFO")
    output = Path(args.output_folder)
    if not args.no_checkpoint:
        rebuild_checkpoint(
            output,
            labels_dirname=args.labels_dirname,
            data_yaml_name=args.data_yaml_name,
            dry_run=args.dry_run,
        )
    if not args.no_flat:
        flat = consolidate_batches(
            output,
            labels_dirname=args.labels_dirname,
            flat_dirname=args.flat_dirname,
            dry_run=args.dry_run,
        )
        drops = [x.strip() for x in str(args.drop_class_names).split(",") if x.strip()]
        if drops:
            # Operate on a copy of data.yaml beside the flat folder so the
            # original batched output stays untouched.
            import shutil as _sh

            flat_yaml = flat.parent / f"{flat.name}.yaml"
            if not args.dry_run:
                _sh.copy2(output / args.data_yaml_name, flat_yaml)
            drop_classes_and_compact(flat, flat_yaml, drops, dry_run=args.dry_run)
    logger.info("Done. Re-run auto_label WITHOUT --no_auto_resume to resume.")


if __name__ == "__main__":
    main()
