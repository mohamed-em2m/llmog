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


def ensure_class_id(class_map, cid, prefix="original_class"):
    """Ensure integer class id ``cid`` exists in ``class_map`` (mutated in place).

    Used by the small-box ``keep`` path, which writes the original YOLO line
    verbatim: without this, a kept box whose id has no name in the map would
    produce a label file referencing a nameless id in data.yaml (untrainable).
    Returns ``(name, added)``; caller must hold the class_map lock when
    threaded. Never remaps an id that is already taken.
    """
    try:
        cid = int(float(str(cid).strip()))
    except (TypeError, ValueError):
        return None, False
    for name, idx in (class_map or {}).items():
        try:
            if int(idx) == cid:
                return name, False
        except (TypeError, ValueError):
            continue
    name = f"{prefix}_{cid}"
    i = 0
    while name in class_map:
        i += 1
        name = f"{prefix}_{cid}_{i}"
    class_map[name] = cid
    return name, True


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
