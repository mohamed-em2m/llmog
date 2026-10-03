"""Write the updated dataset yaml back to disk."""

import yaml
from pathlib import Path
from auto_annotation.logging_utils import logger


def find_renumbered(old_names, class_map):
    """Return [(name, old_id, new_id)] where an existing name would change id.

    Used as a pre-write WARNING so a yaml re-sync never silently renumbers a
    class that finished label files may already reference. The checkpoint is
    always authoritative: callers keep checkpoint ids and only warn here.
    """
    renumbered = []
    if isinstance(old_names, dict):
        old_ids = {str(v): int(k) for k, v in old_names.items()}
    else:
        old_ids = {str(n): i for i, n in enumerate(list(old_names or []))}
    for name, new_id in (class_map or {}).items():
        if name in old_ids and old_ids[name] != int(new_id):
            renumbered.append((name, old_ids[name], int(new_id)))
    return renumbered


def lookup_original_name(orig_names, old_id):
    """Return the input-dataset class name for numeric id ``old_id``.

    ``orig_names`` is the data.yaml ``names`` in file order (list) or an
    id->name mapping (dict); ``None``/unresolvable returns ``None``. Input
    label files only carry numeric ids, so this is how a kept box recovers
    its original label name.
    """
    try:
        old = int(float(str(old_id).strip()))
    except (TypeError, ValueError):
        return None
    if not orig_names:
        return None
    try:
        if isinstance(orig_names, dict):
            for key in (old, str(old)):
                if key in orig_names:
                    name = str(orig_names[key]).strip()
                    return name or None
            return None
        seq = list(orig_names)
        if 0 <= old < len(seq):
            name = str(seq[old]).strip()
            return name or None
    except (TypeError, ValueError, IndexError):
        return None
    return None


def resolve_kept_class(class_map, old_id, prefix="original_class", orig_names=None):
    """Resolve the output class for a kept (never classified) small box.

    Policy: a kept box must never silently merge into an unrelated class,
    and all kept boxes from the same original id must share ONE class.
    - the input name still holds the same id in the current map
      (``orig_names`` from data.yaml) -> keep the ORIGINAL name and id
      verbatim (a true keep; no map mutation, no model call).
    - id free in the map -> keep the ORIGINAL numeric id, registering the
      original name when known, else ``original_class_<id>``.
    - id taken by anything else (convention changed) -> quarantine ONCE
      under ``original_class_<oldid>`` at a FRESH id (max+1); every later
      kept box with the same old id reuses that quarantine instead of
      minting another id. The quarantine name MUST stay synthetic: the
      original name may now mean something else.
    - our own earlier registration (slot or quarantine) -> reuse it.

    NOTE: the ``original_class_<id>`` name pattern is reserved for this
    purpose -- user classes must not use it.

    Returns ``(name, final_id, disposition)`` with disposition in
    {"slot", "reused", "remapped", "original", "invalid"}. Mutates
    ``class_map`` for slot/remapped; caller must hold the class_map lock
    when threaded.
    """
    try:
        old = int(float(str(old_id).strip()))
    except (TypeError, ValueError):
        return None, None, "invalid"
    from auto_annotation.checkpoint import next_free_id as _next_free_id

    class_map = class_map if class_map is not None else {}
    original = lookup_original_name(orig_names, old)
    if original is not None:
        try:
            _holder_id = int(class_map.get(original))
        except (TypeError, ValueError):
            _holder_id = None
        if _holder_id == old:
            return original, old, "original"

    base = f"{prefix}_{old}"
    taken_by = None
    for name, idx in class_map.items():
        try:
            if int(idx) == old:
                taken_by = name
                break
        except (TypeError, ValueError):
            continue
    if base in class_map:
        # Own earlier registration (slot or quarantine) -- reuse it, so
        # repeated keeps never proliferate fresh ids.
        return base, class_map[base], "reused"
    if taken_by is None:
        name = original if (original and original not in class_map) else base
        class_map[name] = old
        return name, old, "slot"
    fresh = _next_free_id(class_map)
    class_map[base] = fresh
    return base, fresh, "remapped"


def save_updated_yaml(yaml_path, output_folder, original_data, class_map):
    if not class_map:
        logger.warning(
            "Class map is empty. Skipping YAML update to prevent erasing existing names."
        )
        return
    renumbered = find_renumbered(original_data.get("names", []), class_map)
    for name, old_id, new_id in renumbered:
        logger.warning(
            f"data.yaml re-sync: class {name!r} moves id {old_id} -> {new_id} "
            "(checkpoint is authoritative; already-written labels use the new id; "
            "continuing -- no labels harmed, but do NOT hand-edit names ordering "
            "mid-run)."
        )
    updated = dict(original_data)
    sorted_names = [name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    updated["names"] = sorted_names
    updated["nc"] = len(sorted_names)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(updated, f, sort_keys=False)

    with open(Path(output_folder) / "data.yaml", "w") as f:
        yaml.safe_dump(updated, f, sort_keys=False)
