from loguru import logger
import sys
from config import LOGS_DIR, LOG_LEVEL, LOG_FORMAT

# Remove default handler
logger.remove()

# Add console handler
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    colorize=True
)

# Add file handler
logger.add(
    LOGS_DIR / "app_{time:YYYY-MM-DD}.log",
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    rotation="00:00",
    retention="7 days",
    compression="zip"
)

def get_logger():
    return logger