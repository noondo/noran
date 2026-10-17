import logging

from production_app.config import LOG_LEVEL


def setup_logging():
    """Configure Python logging once and provide reusable loggers."""
    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name="production_app"):
    return logging.getLogger(name)
