"""Schemes package -- validated configuration objects for every CLI entry point.

The single source of truth is :class:`PipelineConfig` in :mod:`schemes.argument`.
A thin ``Args`` alias is kept so transient callers that still write
``from schemes import Args`` keep working while they migrate.
"""

from schemes.argument import PipelineConfig

# Back-compat alias: legacy callers referred to a single ``Args`` class. New
# code should import :class:`PipelineConfig` directly.
Args = PipelineConfig

__all__ = ["PipelineConfig", "Args"]
