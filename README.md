# wind-up
A tool to assess yield uplift of wind turbines

[![CI](https://github.com/resgroup/wind-up/actions/workflows/CI.yaml/badge.svg)](https://github.com/resgroup/wind-up/actions/workflows/CI.yaml)
[![Python 3.10](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Lint & Format: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy-yellow.svg)](https://github.com/python/mypy)
[![TaskRunner: poethepoet](https://img.shields.io/badge/poethepoet-enabled-1abc9c.svg)](https://github.com/nat-n/poethepoet)

## Getting Started
The wind-up package can be installed in a virtual environment with the following commands:
```shell
# create and activate a virtual environment, if needed
python -m venv .venv
source .venv/Scripts/activate  # or .venv/bin/activate on linux or ".venv/Scripts/activate" in Windows command prompt
# install the wind-up package in the virtual environment
pip install git+https://github.com/resgroup/wind-up.git # alternatively clone the repo, navigate to the wind-up folder and run "pip install ."
```

See [`examples`](examples) folder for example analysis using the wind-up package. [`smarteole_example.ipynb`](examples%2Fsmarteole_example.ipynb) is a good place to start.

## Contributing
To start making changes make a new branch from `main`. Note `main` is protected; 
if a commit fails to push and you want to undo it try `git reset origin/main --hard`

After cloning the repository and (creating and activating the virtual environment), use the following command to install the wind-up package 
in editable mode with the dev dependencies:
```shell
git clone https://github.com/resgroup/wind-up
cd wind-up
# create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate  # or .venv/bin/activate on linux or ".venv/Scripts/activate" in Windows command prompt
# install the package in editable mode with the dev dependencies
pip install -e .[dev] # or .[jupyter,dev] if you want jupyter dependencies as well
```

## Running tests
Install dev dependencies and use `poe test` or `poe all` to run unit tests (make sure the virtual environment is activated)

## License
See [`LICENSE.txt`](LICENSE.txt)

## Contact
Alex.Clerc@res-group.com
