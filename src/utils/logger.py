import logging

_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
}
_RESET = "\033[0m"


class _ColorFormatter(logging.Formatter):
    def format(self, record):
        color = _COLORS.get(record.levelno, "")
        record = logging.makeLogRecord(record.__dict__)
        record.levelname = f"{color}{record.levelname}{_RESET}"
        return super().format(record)


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("flashvsr")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    handler.setFormatter(_ColorFormatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("flashvsr")
