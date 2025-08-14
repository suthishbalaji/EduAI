import logging
import os
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(LOG_LEVEL)

    stream_handler = logging.StreamHandler()
    stream_format = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    stream_handler.setFormatter(stream_format)
    logger.addHandler(stream_handler)

    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_FILE_PATH, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_format = jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger
