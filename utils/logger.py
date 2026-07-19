"""
utils/logger.py

"""

import logging
from logging.handlers import RotatingFileHandler

from config import settings

_LOGGER_NAME = "campus_surveillance"


def get_logger():
    """Return the shared application logger, configuring it on first use."""
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:                       # already configured
        return logger

    settings.ensure_directories()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        settings.LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
