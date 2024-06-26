from pathlib import Path

import pandas as pd

from wind_up.caching import with_parquet_cache


class TestWithParquetCache:
    def test_creates_and_fetches_cache(self, tmp_path: Path) -> None:
        fp = tmp_path / "test.parquet"
        sample_df = pd.DataFrame({"a": [1, 2, 3]})

        @with_parquet_cache(fp=tmp_path / "test.parquet")
        def myfunc() -> pd.DataFrame:
            return sample_df

        assert not fp.exists()
        _df = myfunc()
        pd.testing.assert_frame_equal(_df, sample_df)
        assert fp.exists()

        df2 = myfunc()
        pd.testing.assert_frame_equal(df2, sample_df)

    def test_doesnt_run_the_func_if_file_exists(self, tmp_path: Path) -> None:
        fp = tmp_path / "test.parquet"
        sample_df = pd.DataFrame({"a": [1, 2, 3]})
        sample_df.to_parquet(fp)

        @with_parquet_cache(fp=tmp_path / "test.parquet")
        def myfunc() -> pd.DataFrame:
            return 1 / 0

        _df = myfunc()
        pd.testing.assert_frame_equal(_df, sample_df)
