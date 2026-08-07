"""Process-wide logging configuration for the auto-annotation pipeline."""

import logging

logger = logging.getLogger("auto_annotation")


def setup_logging(log_level="DEBUG", log_file=None):
    """
    Configure logging once, at process start. Always logs to the console;
    additionally logs to --log_file if one is given, so a long unattended
    run can be tailed / grepped / diffed after the fact.
    """
    level = getattr(logging, str(log_level).upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-7s | %(threadName)-12s | %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # avoid duplicate handlers if called more than once

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(file_handler)
        logger.info(f"Logging to console and to file: {log_file}")

    # Quiet down noisy third-party loggers (httpx/openai log every request at INFO).
    logging.getLogger("httpx").setLevel(max(level, logging.WARNING))
    logging.getLogger("httpcore").setLevel(max(level, logging.WARNING))
