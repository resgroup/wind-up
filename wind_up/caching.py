import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from wind_up.result_manager import result_manager


def with_parquet_cache(fp: Path, *, use_cache: bool = True) -> Callable:
    def wrap(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        def wrapped_f(*a: Any, **kw: Any) -> pd.DataFrame:  # noqa
            if not Path(fp).is_file() or not use_cache:
                func(*a, **kw).to_parquet(fp)
            return pd.read_parquet(fp)

        return wrapped_f

    return wrap


def with_pickle_cache(fp: Path, *, use_cache: bool = True) -> Callable:
    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped_f(*a: Any, **kw: Any) -> Any:  # noqa
            fresh_cache = False
            if not Path(fp).is_file() or not use_cache or Path(fp).stat().st_size == 0:
                with Path.open(fp, "wb") as f:
                    pickle.dump(func(*a, **kw), f)
                    fresh_cache = True
            if not fresh_cache:
                result_manager.warning(f"loading cached pickle {fp}")
            with Path.open(fp, "rb") as f:
                return pickle.load(f)

        return wrapped_f

    return wrap
