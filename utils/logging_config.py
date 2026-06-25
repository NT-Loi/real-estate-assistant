from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_system_logging(
    log_file: str | os.PathLike[str] | None = None,
    level: int = logging.INFO,
) -> Path:
    """Configure console + project-wide file logging once."""
    root = Path(__file__).resolve().parent.parent
    path = Path(log_file or os.getenv("SYSTEM_LOG_FILE", "logs/system.log"))
    if not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(level)

    has_console = any(getattr(handler, "_bds_console", False) for handler in logger.handlers)
    if not has_console:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        console.setLevel(level)
        console._bds_console = True
        logger.addHandler(console)

    resolved = path.resolve()
    has_file = any(
        isinstance(handler, logging.FileHandler)
        and Path(getattr(handler, "baseFilename", "")).resolve() == resolved
        for handler in logger.handlers
    )
    if not has_file:
        file_handler = logging.FileHandler(resolved, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return resolved
