"""
Logging configuration for CDSS
"""
import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logging format
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_file = logs_dir / "cdss.log"

# Root logger configuration
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),  # File handler with UTF-8
        logging.StreamHandler(sys.stdout)  # Console handler
    ]
)

# Create logger for CDSS
logger = logging.getLogger("cdss")

# Suppress verbose third-party logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f"cdss.{name}")
