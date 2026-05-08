"""Loguru-compatible logger fallback for standalone evaluation scripts."""

try:
    from loguru import logger
except ImportError:
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    class _Logger:
        def remove(self, *args, **kwargs):
            return None

        def add(self, *args, **kwargs):
            return None

        def debug(self, message, *args, **kwargs):
            logging.debug(message)

        def info(self, message, *args, **kwargs):
            logging.info(message)

        def warning(self, message, *args, **kwargs):
            logging.warning(message)

        def error(self, message, *args, **kwargs):
            logging.error(message)

    logger = _Logger()
