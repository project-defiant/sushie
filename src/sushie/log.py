"""Logging module for sushie."""

from loguru import logger

# Remove default handler to avoid duplicate output
logger.remove()

# Add stderr handler with sushie format
logger.add(
    "stderr",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Add file handler for disk logs
logger.add(
    "logs/sushie_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    level="INFO",
    rotation="1 week",
    compression="gz",
)
