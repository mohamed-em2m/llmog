"""Write the updated dataset yaml back to disk."""

import yaml

from auto_annotation.logging_utils import logger


def save_updated_yaml(yaml_path, output_folder, original_data, class_map):
    if not class_map:
        logger.warning(
            "Class map is empty. Skipping YAML update to prevent erasing existing names."
        )
        return
    updated = dict(original_data)
    sorted_names = [name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    updated["names"] = sorted_names
    updated["nc"] = len(sorted_names)
    with open(yaml_path, "w") as f:
        yaml.safe_dump(updated, f, sort_keys=False)
    with open(output_folder / "data.yaml", "w") as f:
        yaml.safe_dump(updated, f, sort_keys=False)
