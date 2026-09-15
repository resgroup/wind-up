"""Module to manage logging warnings."""

import logging

logger = logging.getLogger(__name__)


class ResultManager:
    """Class to manage results and warnings."""

    stored_warnings: list[str]

    def __init__(self) -> None:
        """Initialize the ResultManager class."""
        self.stored_warnings = []

    def warning(self, msg: str) -> None:
        """Log a warning message and store it in the stored_warnings list."""
        logger.warning(msg)
        self.stored_warnings.append(msg)


result_manager = ResultManager()
