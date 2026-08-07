"""
Back-compat shim.

This file used to be the whole pipeline (1300+ lines). It has been split
into the ``auto_annotation`` package; this module now just re-exports the
public surface so anything that still does ``import auto_annotiontion`` or
``python -m auto_annotiontion`` keeps working unchanged.

New code should import from ``auto_annotation`` directly, e.g.::

    from auto_annotation import main, RunStats, CheckpointManager
"""

import sys

from auto_annotation import *  # noqa: F401,F403  -- re-export shim
from auto_annotation import main


if __name__ == "__main__":
    main()
    # Unreachable in practice (main() exits on error), but keep an explicit
    # Raise-friendly return value path for tools that exec modules.
    sys.exit(0)
