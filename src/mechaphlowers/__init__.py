import logging

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

pd.options.mode.copy_on_write = True


def welcome() -> str:
    """Welcome function to initialize the package

    Args:

    Returns:
        str: some welcoming message
    """
    return "Welcome!"
