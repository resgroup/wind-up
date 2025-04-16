# wind-up
A tool to assess yield uplift of wind turbines

[![image](https://img.shields.io/pypi/v/res-wind-up.svg)](https://pypi.python.org/pypi/res-wind-up)
[![image](https://img.shields.io/pypi/l/res-wind-up.svg)](https://github.com/resgroup/wind-up/blob/main/LICENSE.txt)
[![image](https://img.shields.io/pypi/pyversions/res-wind-up.svg)](https://pypi.python.org/pypi/res-wind-up)
[![Lint & Format: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)
[![Typing: mypy](https://img.shields.io/badge/typing-mypy-yellow.svg)](https://github.com/python/mypy)
[![lint-and-test](https://github.com/resgroup/wind-up/actions/workflows/lint-and-test.yaml/badge.svg)](https://github.com/resgroup/wind-up/actions/workflows/lint-and-test.yaml)

## Getting Started
See [`examples`](examples) folder for example analysis using the wind-up package. [`smarteole_example.ipynb`](examples%2Fsmarteole_example.ipynb) is a good place to start.

The wind-up package can be installed in a virtual environment with the following commands:
```shell
# create and activate a virtual environment, if needed
python -m venv .venv
source .venv/Scripts/activate  # or .venv/bin/activate on linux or ".venv/Scripts/activate" in Windows command prompt
# install the wind-up package in the virtual environment
pip install res-wind-up # alternatively clone the repo, navigate to the wind-up folder and run "pip install ."
```
Note that the package is named `wind_up` (with an underscore) in Python code. For example to print the version of the installed package use the following code snippet:
```python
import wind_up
print(wind_up.__version__)
```

## Contributing
To start making changes fork the repository or make a new branch from `main`. Note `main` is protected; 
if a commit fails to push and you want to undo it try `git reset origin/main --hard`

After cloning the repository (and creating and activating the virtual environment), use the following commands to install the wind-up package in editable mode with the dev dependencies:
```shell
git clone https://github.com/resgroup/wind-up # or your fork of wind-up
cd wind-up
# create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate  # or .venv/bin/activate on linux or ".venv/Scripts/activate" in Windows command prompt
# install the package in editable mode with the dev dependencies
pip install -e .[dev] # or .[all] if you want examples dependencies as well or .[examples] if you want only examples dependencies
```
Use `poe all` to run all required pre-push commands (make sure the virtual environment is activated) or to skip slow 
tests use `poe all-fast`.

## Running tests
Install dev dependencies and use `poe test` to run unit tests (make sure the virtual environment is activated)

For convenience when developing locally, run `poe test-fast` to avoid running the tests marked as slow.

## License
See [`LICENSE.txt`](LICENSE.txt)

## Contact
Alex.Clerc@res-group.com
