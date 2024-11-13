import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def welcome() -> str:
    """Welcome function to initialize the package

    Args:

    Returns:
        str: some welcoming message
    """
    return "Welcome!"
