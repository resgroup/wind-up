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

The wind-up package can be installed using any Python environment manager. Examples using [uv](https://docs.astral.sh/uv/) and [pip](https://pypi.org/project/pip/) are shown below:

### Using uv
```shell
# Install the wind-up package in an existing project
uv add res-wind-up

# Or create a new project and install the wind-up package
uv init my-project
cd my-project
uv add res-wind-up
```

### Using pip
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

The development environment should be created and managed using [uv](https://docs.astral.sh/uv/). To create the environment:
```shell
uv sync --extra dev
```
To run the formatting, linting and testing:
```shell
uv run poe all # or all-fast to skip slow tests
```
Or simply
```shell
poe all # or all-fast to skip slow tests
```
if you have activated the virtual environment.

## License
See [`LICENSE.txt`](LICENSE.txt)

## Contact
Alex.Clerc@res-group.com
