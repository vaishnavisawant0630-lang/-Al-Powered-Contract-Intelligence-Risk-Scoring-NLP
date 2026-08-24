"""
core/
=====
The foundation layer of the Contract Intelligence platform.

This package is the ONLY package that every other package is allowed to import from.
It defines shared types, configuration, logging, and exceptions.

DEPENDENCY RULE: core imports from NOTHING inside this project.
All other packages (ingestion, data_processing, ner, api) import FROM core.
No sibling package may import from another sibling — only from core.

Exports (public API of this package):
    - get_settings()        → validated app-wide Settings object
    - get_logger(name)      → configured structlog BoundLogger
    - types                 → all shared dataclasses / enums
    - exceptions            → full typed exception hierarchy
"""

from core.config import get_settings
from core.logging import get_logger

__all__ = ["get_settings", "get_logger"]
