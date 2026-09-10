"""Offline tests for the Hill of Towie source adapter.

These cover the pure SCADA transforms: the source-native wide-to-long reshape
(:func:`scada_wide_to_long`) the rest of the pipeline sees, and the v0-only wind-up-format
on-ramp (:func:`long_to_wind_up_format`) that aliases the columns and derives ``PitchAngleMean``
/ ``ShutdownDuration``. The network path (Zenodo download via ``load_hot_scada``) is exercised
separately by a ``slow``-marked test that is excluded from the default offline suite.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from benchmarking.synthetic import HOT_COLUMNS
from benchmarking.synthetic.sources.hill_of_towie import (
    download_zenodo_data,
    ensure_hot_data_files,
    long_to_wind_up_format,
    scada_wide_to_long,
)
from wind_up_v0.constants import TIMESTAMP_COL, DataColumns

if TYPE_CHECKING:
    from pathlib import Path

TIMEBASE_S = 600

# Source-native 10-min tag names, as the loader emits them before any v0 aliasing.
_F_ACTIVE_POWER = "wtc_ActPower_mean"
_F_ACTIVE_POWER_SD = "wtc_ActPower_stddev"
_F_WIND_SPEED = "wtc_AcWindSp_mean"
_F_WIND_SPEED_SD = "wtc_AcWindSp_stddev"
_F_YAW_MEAN = "wtc_NacelPos_mean"
_F_YAW_MIN = "wtc_NacelPos_min"
_F_YAW_MAX = "wtc_NacelPos_max"
_F_GEN_RPM = "wtc_GenRpm_mean"
_F_PITCH_A = "wtc_PitcPosA_mean"
_F_PITCH_B = "wtc_PitcPosB_mean"
_F_PITCH_C = "wtc_PitcPosC_mean"
_F_TIME_READY = "wtc_ScReToOp_timeon"


def _wide_scada_df(*, turbines: tuple[str, ...] = ("T01", "T02"), periods: int = 6) -> pd.DataFrame:
    """Build a fabricated wide two-level SCADA frame as ``load_hot_10min_data`` emits it.

    Level 0 of the columns is the turbine name (index name ``StationId``); level 1 is the
    source-native ``wtc_*`` tag name. The row index is the start-format timestamp.
    """
    index = pd.date_range("2020-01-01", periods=periods, freq="10min", tz="UTC")
    index.name = TIMESTAMP_COL

    fields = {
        _F_ACTIVE_POWER: np.linspace(100.0, 600.0, periods),
        _F_ACTIVE_POWER_SD: np.full(periods, 10.0),
        _F_WIND_SPEED: np.linspace(5.0, 10.0, periods),
        _F_WIND_SPEED_SD: np.full(periods, 1.0),
        _F_YAW_MEAN: np.full(periods, 180.0),
        _F_YAW_MIN: np.full(periods, 175.0),
        _F_YAW_MAX: np.full(periods, 185.0),
        _F_GEN_RPM: np.linspace(1000.0, 1500.0, periods),
        _F_PITCH_A: np.full(periods, 1.0),
        _F_PITCH_B: np.full(periods, 2.0),
        _F_PITCH_C: np.full(periods, 3.0),
        _F_TIME_READY: np.full(periods, float(TIMEBASE_S)),
    }
    # Both turbines share identical values at each timestamp but vary over time. Stuck
    # detection must compare each turbine to its OWN previous record (grouped by turbine),
    # so two turbines that merely match each other are not mistaken for frozen data.
    per_turbine = {turbine: pd.DataFrame(fields, index=index) for turbine in turbines}
    wide = pd.concat(per_turbine, axis=1)
    wide.columns = wide.columns.set_names(["StationId", None])
    return wide


def test_scada_wide_to_long_emits_source_native_long_format() -> None:
    """The wide two-level frame becomes a single-level long frame keyed by the source schema."""
    long_df = scada_wide_to_long(_wide_scada_df())

    assert long_df.columns.nlevels == 1
    assert long_df.index.name == TIMESTAMP_COL
    assert set(long_df[HOT_COLUMNS.turbine].unique()) == {"T01", "T02"}
    for col in (HOT_COLUMNS.active_power, HOT_COLUMNS.wind_speed, HOT_COLUMNS.wind_speed_sd, HOT_COLUMNS.gen_rpm):
        assert col in long_df.columns
    # The v0-only derived columns are not added by the source-native reshape.
    assert DataColumns.pitch_angle_mean not in long_df.columns
    assert DataColumns.shutdown_duration not in long_df.columns


def test_long_to_wind_up_format_aliases_and_computes_mean_pitch() -> None:
    """The v0 on-ramp aliases the source columns and derives PitchAngleMean."""
    wind_up_df = long_to_wind_up_format(scada_wide_to_long(_wide_scada_df()))

    assert DataColumns.active_power_mean in wind_up_df.columns
    assert np.allclose(wind_up_df[DataColumns.pitch_angle_mean], 2.0)


def test_calc_shutdown_duration_zero_when_fully_available() -> None:
    """Two turbines that share values per timestamp but vary over time are all available.

    Stuck detection compares each turbine to its own previous record, so turbines that
    merely match each other at the same timestamp are not flagged as frozen.
    """
    wind_up_df = long_to_wind_up_format(scada_wide_to_long(_wide_scada_df()))
    assert DataColumns.shutdown_duration in wind_up_df.columns
    assert np.allclose(wind_up_df[DataColumns.shutdown_duration], 0.0)


def test_calc_shutdown_duration_flags_stuck_data() -> None:
    """Repeated (stuck) rows above the low-wind threshold are flagged as full downtime."""
    wide = _wide_scada_df(turbines=("T01",), periods=4)
    # Freeze T01 to identical values across all rows (stuck), at a wind speed above 1.5 m/s.
    for field in wide.columns.get_level_values(1).unique():
        wide.loc[:, ("T01", field)] = wide.iloc[0].loc[("T01", field)]
    wind_up_df = long_to_wind_up_format(scada_wide_to_long(wide))
    # First row has no prior to diff against; subsequent stuck rows are full downtime.
    assert np.allclose(wind_up_df[DataColumns.shutdown_duration].iloc[1:], TIMEBASE_S)


def test_calc_shutdown_duration_flags_only_the_frozen_turbine() -> None:
    """In a multi-turbine frame, only the turbine whose own data is frozen is flagged."""
    wide = _wide_scada_df(turbines=("T01", "T02"), periods=4)
    # Freeze T01 across all rows (stuck); T02 keeps varying over time.
    for field in wide.columns.get_level_values(1).unique():
        wide.loc[:, ("T01", field)] = wide.iloc[0].loc[("T01", field)]
    wind_up_df = long_to_wind_up_format(scada_wide_to_long(wide))
    t01 = wind_up_df.loc[wind_up_df[DataColumns.turbine_name] == "T01", DataColumns.shutdown_duration].to_numpy()
    t02 = wind_up_df.loc[wind_up_df[DataColumns.turbine_name] == "T02", DataColumns.shutdown_duration].to_numpy()
    assert np.allclose(t01[1:], TIMEBASE_S)  # frozen turbine -> downtime after the first row
    assert np.allclose(t02, 0.0)  # varying turbine -> available throughout


def test_download_routes_through_one_session_closed_once(tmp_path: Path) -> None:
    """Every Zenodo request goes through a single Session that is closed exactly once.

    Regression guard for leaked SSL sockets: a per-call ``requests.get`` closes its
    transient connection pool before the streamed response's socket is released back to
    it, so the socket lingers until GC and trips ``filterwarnings = error`` via
    ResourceWarning. A single Session whose pool is closed on exit releases every socket
    deterministically. Asserting one Session, closed once, encodes exactly that fix.
    """
    # Pre-cache the metadata so this stays offline; the file-download branch (the leak
    # source) still has to route its streamed get through the shared Session.
    file_entry = {"key": "small.txt", "size": 10, "links": {"self": "https://zenodo.test/small.txt"}}
    (tmp_path / "zenodo_dataset_metadata.json").write_text(json.dumps({"files": [file_entry]}))

    response = MagicMock()
    response.status_code = 200
    response.iter_content.return_value = [b"0123456789"]
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    session = MagicMock()
    session.get.return_value = response
    session.__enter__.return_value = session
    session.__exit__.return_value = False

    def _banned_get(*_args: object, **_kwargs: object) -> object:
        msg = "per-call requests.get leaks sockets; route every request through the shared Session"
        raise AssertionError(msg)

    with (
        patch("requests.Session", return_value=session) as session_cls,
        patch("requests.get", side_effect=_banned_get),
    ):
        download_zenodo_data(record_id="123", output_dir=tmp_path)

    session_cls.assert_called_once()  # exactly one Session for the whole download
    session.__enter__.assert_called_once()
    session.__exit__.assert_called_once()  # pool (and its sockets) closed deterministically
    session.get.assert_called_once_with(
        file_entry["links"]["self"],
        stream=True,
        timeout=(10, 60),
        headers={},
    )
    assert (tmp_path / "small.txt").read_bytes() == b"0123456789"


_BIG_FILE_ENTRY = {"key": "big.bin", "size": 10, "links": {"self": "https://zenodo.test/big.bin"}}
_ZIP_FILE_ENTRY = {"key": "2017.zip", "size": 10, "links": {"self": "https://zenodo.test/2017.zip"}}


def _stub_response(*, status_code: int = 200, chunks: list[bytes] | None = None) -> MagicMock:
    """A context-manager ``requests.Response`` stub streaming ``chunks``."""
    response = MagicMock()
    response.status_code = status_code
    response.iter_content.return_value = list(chunks or [])
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    if status_code >= 400:
        error = requests.HTTPError(f"{status_code} Error", response=response)
        response.raise_for_status.side_effect = error
    else:
        response.raise_for_status.return_value = None
    return response


def _run_download(tmp_path: Path, *, file_entry: dict, responses: list[MagicMock]) -> MagicMock:
    """Run ``download_zenodo_data`` offline against a scripted sequence of responses."""
    (tmp_path / "zenodo_dataset_metadata.json").write_text(json.dumps({"files": [file_entry]}))

    session = MagicMock()
    session.get.side_effect = responses
    session.__enter__.return_value = session
    session.__exit__.return_value = False

    with patch("requests.Session", return_value=session), patch("time.sleep"):
        download_zenodo_data(record_id="123", output_dir=tmp_path)
    return session


class TestZenodoDownloadRetries:
    """A required Zenodo file survives the transient failures the real service returns."""

    file_entry = _BIG_FILE_ENTRY

    def test_gateway_timeout_is_retried_then_succeeds(self, tmp_path: Path) -> None:
        session = _run_download(
            tmp_path,
            file_entry=self.file_entry,
            responses=[_stub_response(status_code=504), _stub_response(chunks=[b"0123456789"])],
        )
        assert session.get.call_count == 2
        assert (tmp_path / "big.bin").read_bytes() == b"0123456789"

    def test_a_truncated_stream_is_retried_and_resumed(self, tmp_path: Path) -> None:
        session = _run_download(
            tmp_path,
            file_entry=self.file_entry,
            responses=[
                _stub_response(chunks=[b"01234"]),  # stream ends 5 bytes short
                _stub_response(status_code=206, chunks=[b"56789"]),
            ],
        )
        assert session.get.call_count == 2
        # the second attempt resumed from the bytes already on disk
        assert session.get.call_args_list[1].kwargs["headers"] == {"Range": "bytes=5-"}
        assert (tmp_path / "big.bin").read_bytes() == b"0123456789"

    def test_an_oversized_file_is_discarded_and_downloaded_afresh(self, tmp_path: Path) -> None:
        """A resume that re-appended bytes leaves a longer file, which is corrupt, not complete."""
        (tmp_path / "big.bin").write_bytes(b"0123456789EXTRA")

        session = _run_download(
            tmp_path, file_entry=self.file_entry, responses=[_stub_response(chunks=[b"0123456789"])]
        )
        assert session.get.call_args_list[0].kwargs["headers"] == {}  # no Range: restarted from 0
        assert (tmp_path / "big.bin").read_bytes() == b"0123456789"

    def test_a_client_error_is_not_retried_and_closes_its_response(self, tmp_path: Path) -> None:
        response = _stub_response(status_code=404)
        with pytest.raises(requests.HTTPError):
            _run_download(tmp_path, file_entry=self.file_entry, responses=[response])

        # closed even though raise_for_status() failed, so the socket does not linger to GC
        response.__exit__.assert_called_once()


class TestEnsureHotDataFiles:
    """The cache-hit shortcut only fires for files that are actually whole."""

    file_entry = _ZIP_FILE_ENTRY

    def _write_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "zenodo_dataset_metadata.json").write_text(json.dumps({"files": [self.file_entry]}))

    def test_a_complete_file_makes_no_network_call(self, tmp_path: Path) -> None:
        self._write_metadata(tmp_path)
        (tmp_path / "2017.zip").write_bytes(b"0123456789")

        with patch("benchmarking.synthetic.sources.hill_of_towie.download_zenodo_data") as download:
            ensure_hot_data_files(["2017.zip"], data_dir=tmp_path)

        download.assert_not_called()

    def test_an_oversized_file_is_re_downloaded(self, tmp_path: Path) -> None:
        self._write_metadata(tmp_path)
        (tmp_path / "2017.zip").write_bytes(b"0123456789EXTRA")  # a resume re-appended bytes

        with patch("benchmarking.synthetic.sources.hill_of_towie.download_zenodo_data") as download:
            ensure_hot_data_files(["2017.zip"], data_dir=tmp_path)

        assert download.call_args.kwargs["filenames"] == ["2017.zip"]

    def test_a_truncated_file_is_re_downloaded(self, tmp_path: Path) -> None:
        self._write_metadata(tmp_path)
        (tmp_path / "2017.zip").write_bytes(b"01234")  # interrupted download left it short

        with patch("benchmarking.synthetic.sources.hill_of_towie.download_zenodo_data") as download:
            ensure_hot_data_files(["2017.zip"], data_dir=tmp_path)

        assert download.call_args.kwargs["filenames"] == ["2017.zip"]
