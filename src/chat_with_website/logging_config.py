"""Central logging setup. Call `setup_logging()` once at program start."""

from __future__ import annotations

import logging
import sys

from chat_with_website.config import settings

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str | None = None) -> None:
    root = logging.getLogger()
    if root.handlers:  # already configured (e.g. Streamlit reruns)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel((level or settings.log_level).upper())

    # Quieten chatty libraries
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
