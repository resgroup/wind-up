# wind-up
A tool to assess yield uplift of wind turbines

[![CI](https://github.com/resgroup/wind-up/actions/workflows/push_or_pr.yaml/badge.svg)](https://github.com/resgroup/wind-up/actions/workflows/push_or_pr.yaml)
[![Python 3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Lint & Format: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy-yellow.svg)](https://github.com/python/mypy)
[![TaskRunner: poethepoet](https://img.shields.io/badge/poethepoet-enabled-1abc9c.svg)](https://github.com/nat-n/poethepoet)

## Getting Started
Currently, the package has to be installed locally. To do this, clone the repository and navigate to the root folder.

Then create a virtual environment and install the package in it:
```shell
# create a virtual environment in the .venv folder
python -m venv .venv
# activate the virtual environment
source .venv/Scripts/activate  # or .venv/bin/activate on linux
# install the wind-up package
pip install -r requirements.txt
```

See `examples` folder for analysis examples

## Contributing
To start making changes make a new branch from `main`. Note `main` is protected; 
if a commit fails to push and you want to undo it try `git reset origin/main --hard`

after creating and activating the environment as above, use the following command to install the package 
in editable mode with the dev dependencies:
```shell
# install the package in editable mode with the dev dependencies
pip install -r dev-requirements.txt -e .
```

## Running tests
Use `poe test` or `poe all` to run unit tests (make sure the virtual environment is activated)

## Updating and locking dependencies
To update the main (and dev) dependencies based off the `pyproject.toml` file conditions use:
```shell
pip-compile -o requirements.txt pyproject.toml
pip-compile --extra dev -o dev-requirements.txt pyproject.toml
```

Then you can use the following to update the dependencies in the virtual environment:
```shell
pip install -r dev-requirements.txt   
# or -r requirements.txt if only using main dependencies
```

## Contact
Alex.Clerc@res-group.com
