"""``auto_annotation`` package -- relabel binary defect/no-defect YOLO
annotations into multi-class defect labels using a vision-language model.

This is the split-up version of the former single-file ``auto_annotiontion.py``.
Importing this package re-exports the public surface that downstream callers
and tests are most likely to want, so e.g.::

    from auto_annotation import main, RunStats, CheckpointManager

Two latent bugs in the original file are fixed here:

* ``servers/__init__.py`` exports ``servers_factory`` (not ``factory``);
  ``auto_annotation.server_init`` uses the correct name and also keeps a
  ``factory`` alias for any external caller that still depends on the old
  name.
* The original ``build_client`` called ``init_server(args.server_type, args)``
  with two args, but ``init_server`` only takes ``args``. The two-arg call
  would have raised ``TypeError`` at runtime for any non-external server.
"""

from .logging_utils import setup_logging, logger
from .stats import RunStats
from .checkpoint import CheckpointManager
from .server_init import (
    wait_for_server_health,
    init_server,
    init_vllm_server,
    build_client,
)
from .image_io import (
    encode_crop_to_data_uri,
    detect_defect,
    load_or_init_class_map,
    find_labeled_images,
    chunk_list,
)
from .single_image import process_one_image
from .batch_runner import read_images_with_labels
from .yaml_utils import save_updated_yaml
from .cli import parse_args
from .main import main


__all__ = [
    "setup_logging",
    "logger",
    "RunStats",
    "CheckpointManager",
    "wait_for_server_health",
    "init_server",
    "init_vllm_server",
    "build_client",
    "encode_crop_to_data_uri",
    "detect_defect",
    "load_or_init_class_map",
    "find_labeled_images",
    "chunk_list",
    "process_one_image",
    "read_images_with_labels",
    "save_updated_yaml",
    "parse_args",
    "main",
]
