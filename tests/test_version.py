from pathlib import Path

import toml

import wind_up
from wind_up.constants import PROJECTROOT_DIR


def test_versions_match() -> None:
    with Path.open(PROJECTROOT_DIR / "pyproject.toml") as toml_file:
        pyproject_data = toml.load(toml_file)
    package_version_from_toml = pyproject_data["project"]["version"]
    package_version_from_init = wind_up.__version__
    assert package_version_from_toml == package_version_from_init
