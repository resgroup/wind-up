import os
from pathlib import Path
from typing import IO, Any

import yaml


class Loader(yaml.SafeLoader):
    def __init__(self: "Loader", stream: IO) -> None:
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

    @property
    def root(self: "Loader") -> str:
        return self._root


def construct_include(loader: Loader, node: yaml.Node) -> Any:  # noqa ANN401
    filename = Path.resolve(Path(loader.root) / Path(loader.construct_scalar(node)))  # type: ignore[arg-type]
    extension = os.path.splitext(filename)[1].lstrip(".")  # noqa PTH122

    with Path.open(filename) as f:
        return yaml.load(f, Loader) if extension in ("yaml", "yml") else "".join(f.readlines())  # noqa S506
