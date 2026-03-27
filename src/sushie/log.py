"""Logging module for sushie."""

import logging

logger = logging.getLogger("sushie")
logger.setLevel(logging.INFO)

# Prevent propagating to root logger
logger.propagate = False
