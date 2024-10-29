import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def welcome() -> str:
    return "Welcome!"
