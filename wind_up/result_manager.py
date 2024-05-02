import logging

logger = logging.getLogger(__name__)


class ResultManager:
    _warnings: list[str]

    def __init__(self) -> None:
        self._warnings = []

    def warning(self, msg: str) -> None:
        logger.warning(msg)
        self._warnings.append(msg)


result_manager = ResultManager()
