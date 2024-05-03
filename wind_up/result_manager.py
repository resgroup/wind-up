import logging

logger = logging.getLogger(__name__)


class ResultManager:
    stored_warnings: list[str]

    def __init__(self) -> None:
        self.stored_warnings = []

    def warning(self, msg: str) -> None:
        logger.warning(msg)
        self.stored_warnings.append(msg)


result_manager = ResultManager()
