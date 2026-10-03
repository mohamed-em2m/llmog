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


def resolve_kept_class(class_map, old_id, prefix="original_class"):
    """Resolve the output class for a kept (never classified) small box.

    Policy: a kept box must never silently merge into an unrelated class,
    and all kept boxes from the same original id must share ONE class.
    - id free in the map -> keep the ORIGINAL numeric id, registering
      ``original_class_<id>`` (slot preserved, tidy map).
    - id taken by anything else (e.g. cross-convention relabeling where old
      binary id 1 now means something else) -> quarantine ONCE under
      ``original_class_<oldid>`` at a FRESH id (max+1); every later kept
      box with the same old id reuses that quarantine instead of minting
      another id.
    - our own earlier registration (slot or quarantine) -> reuse it.

    NOTE: the ``original_class_<id>`` name pattern is reserved for this
    purpose -- user classes must not use it. A pre-existing entry with that
    name is treated as our quarantine and reused.

    Returns ``(name, final_id, disposition)`` with disposition in
    {"slot", "reused", "remapped", "invalid"}. Mutates ``class_map`` for
    slot/remapped; caller must hold the class_map lock when threaded.
    """
    try:
        old = int(float(str(old_id).strip()))
    except (TypeError, ValueError):
        return None, None, "invalid"
    from auto_annotation.checkpoint import next_free_id as _next_free_id

    base = f"{prefix}_{old}"
    taken_by = None
    for name, idx in (class_map or {}).items():
        try:
            if int(idx) == old:
                taken_by = name
                break
        except (TypeError, ValueError):
            continue
    if base in (class_map or {}):
        # Own earlier registration (slot or quarantine) -- reuse it, so
        # repeated keeps never proliferate fresh ids.
        return base, class_map[base], "reused"
    if taken_by is None:
        class_map[base] = old
        return base, old, "slot"
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
