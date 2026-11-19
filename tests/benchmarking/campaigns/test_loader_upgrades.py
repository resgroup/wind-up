"""Tests for declarations that list their upgrades and analyses, and for the renamed keys."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from benchmarking.campaigns.declared import DeclaredAnalysis, DeclaredUpgrade, campaign_spec
from benchmarking.campaigns.loader import SHORTHAND_UPGRADE, load_declaration
from benchmarking.synthetic import ToggleSchedule
from wind_up.labels import NORTHING, UPLIFT
from wind_up.wake_reach import MAX_REACH_D, reach_from_uplift

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.campaigns.loader import Declaration

TURBINES_CSV = """Name,Latitude,Longitude,rotor_diameter_m
T01,57.40,-3.30,82
T02,57.60,-3.20,82
T03,57.50,-3.25,82
T04,57.51,-3.25,82
T05,57.52,-3.24,82
"""

# T01 to T04 received the AeroUp over their own works; T01 and T02 then the TuneUp.
AEROUP_CSV = """Turbine,First date of works,Last date of works
T01,2017-03-01,2017-03-03
T02,2017-04-01,2017-04-02
T03,2017-05-01,2017-05-04
T04,2017-06-01,2017-06-02
"""

HEAD = """
name: demo

data:
  scada: scada.parquet
  schema: hill_of_towie
  turbines: turbines.csv

turbines:
  excluded: [T05]
  rated_power_kw: 2050
"""

UPGRADES = """
upgrades:
  - name: AeroUp
    installation_windows: aeroup.csv
    expected_uplift: 0.05
  - name: TuneUp
    turbines: [T01, T02]
    timing: {mode: prepost, changeover: 2018-03-14T18:00:00Z}
    installation_windows:
      - {turbine: T01, start: 2018-03-14T07:00:00Z, end: 2018-03-14T18:00:00Z}
      - {turbine: T02, start: 2018-03-14T07:00:00Z, end: 2018-03-14T18:00:00Z}
    expected_uplift: 0.01
"""

ANALYSES = """
analyses:
  - name: TuneUp
    upgrade: TuneUp
"""

# The TuneUp as a single-upgrade declaration: every works window in one table, each at its own reach.
SHORTHAND_CSV = """Turbine,First date of works,Last date of works,Expected uplift
T01,2017-03-01,2017-03-03,0.05
T02,2017-04-01,2017-04-02,0.05
T03,2017-05-01,2017-05-04,0.05
T04,2017-06-01,2017-06-02,0.05
T01,2018-03-14T07:00:00Z,2018-03-14T18:00:00Z,0.01
T02,2018-03-14T07:00:00Z,2018-03-14T18:00:00Z,0.01
"""

SHORTHAND = """
name: demo

data:
  scada: scada.parquet
  schema: hill_of_towie
  turbines: turbines.csv
  installation_windows: shorthand.csv

turbines:
  upgraded: [T01, T02]
  excluded: [T05]
  rated_power_kw: 2050

upgrade:
  expected_uplift: 0.01

timing:
  mode: prepost
  changeover: 2018-03-14T18:00:00Z
"""

TOGGLE = """
upgrades:
  - name: steering
    turbines: [T01, T02]
    timing: {mode: toggle, start: 2018-01-01T00:00:00Z, period: 2h}
    settling_time: 10min
    wake_reach_d: 6
analyses:
  - name: steering
    upgrade: steering
    period: {start: 2018-01-01T00:00:00Z, end: 2018-07-01T00:00:00Z}
"""


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def load(tmp_path: Path, text: str) -> Declaration:
    """Write the sidecar files and the declaration, and load it."""
    (tmp_path / "turbines.csv").write_text(TURBINES_CSV)
    (tmp_path / "scada.parquet").write_bytes(b"")
    (tmp_path / "aeroup.csv").write_text(AEROUP_CSV)
    (tmp_path / "shorthand.csv").write_text(SHORTHAND_CSV)
    path = tmp_path / "campaign.yaml"
    path.write_text(textwrap.dedent(text))
    return load_declaration(path)


def load_upgrades(
    tmp_path: Path, *, upgrades: str = UPGRADES, analyses: str = ANALYSES, extra: str = ""
) -> Declaration:
    return load(tmp_path, HEAD + upgrades + analyses + extra)


class TestTwoUpgrades:
    def test_the_upgrades_are_read_in_order(self, tmp_path: Path) -> None:
        declaration = load_upgrades(tmp_path)
        aeroup, tuneup = declaration.upgrades
        assert aeroup.name == "AeroUp"
        assert aeroup.turbines == ("T01", "T02", "T03", "T04")
        assert aeroup.timing is None
        assert aeroup.installation_windows["T03"] == [(utc("2017-05-01"), utc("2017-05-05"))]
        assert aeroup.wake_reach_d == pytest.approx(reach_from_uplift(0.05))
        assert tuneup.timing == utc("2018-03-14 18:00")
        assert tuneup.installation_windows["T01"] == [(utc("2018-03-14 07:00"), utc("2018-03-14 18:00"))]
        assert not declaration.shorthand

    def test_the_analysis_measures_the_upgrades_turbines_against_the_rest(self, tmp_path: Path) -> None:
        spec = load_upgrades(tmp_path).spec_for("TuneUp")
        assert spec.upgraded_turbines == ["T01", "T02"]
        # the turbines with only the AeroUp are references; the excluded one is not
        assert spec.candidate_references == ["T03", "T04"]
        assert spec.excluded_turbines == ["T05"]
        assert not spec.references_declared
        assert spec.timing_for("T01") == utc("2018-03-14 18:00")
        assert spec.wake_reach_d == pytest.approx(reach_from_uplift(0.01))

    def test_every_upgrades_windows_are_works_at_their_own_reach(self, tmp_path: Path) -> None:
        spec = load_upgrades(tmp_path).spec_for("TuneUp")
        aeroup = (utc("2017-03-01"), utc("2017-03-04"))
        deployment = (utc("2018-03-14 07:00"), utc("2018-03-14 18:00"))
        assert spec.works["T01"] == [aeroup, deployment]
        assert spec.reach_of("T01", aeroup) == pytest.approx(reach_from_uplift(0.05))
        assert spec.reach_of("T01", deployment) == pytest.approx(reach_from_uplift(0.01))
        assert spec.changeovers()["T03"] == [utc("2017-05-05")]

    def test_it_is_the_campaign_the_single_upgrade_declaration_gives(self, tmp_path: Path) -> None:
        listed = load_upgrades(tmp_path).spec
        short = load(tmp_path, SHORTHAND).spec
        for name in (
            "upgraded_turbines",
            "upgrade_timing",
            "candidate_references",
            "excluded_turbines",
            "north_offsets",
            "rated_power_kw",
            "analysis_period",
            "turbine_col",
            "works",
            "exclusions",
            "operating_state_overrides",
            "references_declared",
            "wake_reach_d",
            "settling_time",
        ):
            assert getattr(listed, name) == getattr(short, name), name
        for turbine, windows in short.works.items():
            for window in windows:
                assert listed.reach_of(turbine, window) == pytest.approx(short.reach_of(turbine, window))
        assert listed.coords == short.coords

    def test_the_one_analysis_is_the_declarations_spec(self, tmp_path: Path) -> None:
        declaration = load_upgrades(tmp_path)
        assert declaration.spec.upgraded_turbines == ["T01", "T02"]
        assert declaration.expected_uplift == 0.01

    def test_subjects_references_and_a_period_can_be_declared(self, tmp_path: Path) -> None:
        analyses = """
analyses:
  - name: T01 only
    upgrade: TuneUp
    subjects: [T01]
    references: [T03]
    period: {start: 2018-01-01T00:00:00Z, end: 2018-06-01T00:00:00Z}
    report: campaign
"""
        declaration = load_upgrades(tmp_path, analyses=analyses)
        (analysis,) = declaration.analyses
        assert analysis.report == ("campaign",)
        spec = declaration.spec_for("T01 only")
        assert spec.upgraded_turbines == ["T01"]
        assert spec.candidate_references == ["T03"]
        assert spec.references_declared
        assert spec.analysis_period == (utc("2018-01-01"), utc("2018-06-01"))

    def test_a_period_per_subject_can_be_declared(self, tmp_path: Path) -> None:
        analyses = """
analyses:
  - name: TuneUp
    upgrade: TuneUp
    period:
      T01: {start: 2018-01-01T00:00:00Z, end: 2018-06-01T00:00:00Z}
"""
        spec = load_upgrades(tmp_path, analyses=analyses).spec_for("TuneUp")
        assert spec.period_for("T01") == (utc("2018-01-01"), utc("2018-06-01"))
        assert spec.period_for("T02") is None

    def test_several_analyses_each_have_a_campaign(self, tmp_path: Path) -> None:
        analyses = """
analyses:
  - {name: first, upgrade: TuneUp, subjects: [T01]}
  - {name: second, upgrade: TuneUp, subjects: [T02]}
"""
        declaration = load_upgrades(tmp_path, analyses=analyses)
        assert declaration.spec_for("first").upgraded_turbines == ["T01"]
        assert declaration.spec_for("second").upgraded_turbines == ["T02"]
        with pytest.raises(ValueError, match="2 analyses"):
            _ = declaration.spec
        with pytest.raises(KeyError, match="third"):
            declaration.spec_for("third")

    def test_the_resolved_echo_lists_the_upgrades_and_analyses(self, tmp_path: Path) -> None:
        resolved = load_upgrades(tmp_path).resolved()
        assert [u["name"] for u in resolved["upgrades"]] == ["AeroUp", "TuneUp"]
        assert resolved["upgrades"][1]["timing"] == {"mode": "prepost", "changeover": "2018-03-14 18:00:00+00:00"}
        assert resolved["upgrades"][0]["installation_windows"]["T04"] == [
            ["2017-06-01 00:00:00+00:00", "2017-06-03 00:00:00+00:00", pytest.approx(reach_from_uplift(0.05))]
        ]
        assert resolved["analyses"] == [
            {
                "name": "TuneUp",
                "upgrade": "TuneUp",
                "subjects": ["T01", "T02"],
                "references": ["T03", "T04"],
                "period": "chosen per upgraded turbine",
                "include_labels": [],
                "exclude_labels": [],
                "report": ["aep", "campaign"],
            }
        ]
        assert resolved["turbines"] == {"excluded": ["T05"], "rated_power_kw": 2050.0}
        assert "timing" not in resolved


class TestAToggleUpgrade:
    def test_its_schedule_and_settling_time_reach_the_campaign(self, tmp_path: Path) -> None:
        declaration = load(tmp_path, HEAD + TOGGLE)
        (upgrade,) = declaration.upgrades
        assert upgrade.settling_time == pd.Timedelta(minutes=10)
        spec = declaration.spec
        assert spec.upgrade_timing == ToggleSchedule(period=pd.Timedelta(hours=2), start=utc("2018-01-01"))
        assert spec.settling_time == pd.Timedelta(minutes=10)
        assert spec.wake_reach_d == 6.0
        resolved = declaration.resolved()["upgrades"][0]
        assert resolved["settling_time"] == "0 days 00:10:00"
        assert resolved["timing"]["mode"] == "toggle"

    def test_the_single_upgrade_keys_take_a_settling_time(self, tmp_path: Path) -> None:
        text = SHORTHAND.replace(
            "timing:\n  mode: prepost\n  changeover: 2018-03-14T18:00:00Z",
            "timing:\n  mode: toggle\n  start: 2018-01-01T00:00:00Z\n  period: 2h",
        ).replace("expected_uplift: 0.01", "expected_uplift: 0.01\n  settling_time: 20min")
        text = text.replace("  installation_windows: shorthand.csv\n", "")
        text += "analysis_period:\n  start: 2018-01-01T00:00:00Z\n  end: 2018-07-01T00:00:00Z\n"
        assert load(tmp_path, text).spec.settling_time == pd.Timedelta(minutes=20)

    def test_a_prepost_upgrade_takes_no_settling_time(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="settling time"):
            load(tmp_path, SHORTHAND.replace("expected_uplift: 0.01", "expected_uplift: 0.01\n  settling_time: 1h"))
        upgrades = UPGRADES.replace("    expected_uplift: 0.01", "    expected_uplift: 0.01\n    settling_time: 1h")
        with pytest.raises(ValueError, match="settling time"):
            load_upgrades(tmp_path, upgrades=upgrades)

    def test_a_negative_settling_time_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="negative"):
            load(tmp_path, HEAD + TOGGLE.replace("settling_time: 10min", "settling_time: -10min"))


class TestTheReach:
    def test_neither_expected_uplift_nor_reach_gives_the_default(self, tmp_path: Path) -> None:
        upgrades = UPGRADES.replace("    expected_uplift: 0.01\n", "")
        assert load_upgrades(tmp_path, upgrades=upgrades).spec.wake_reach_d == MAX_REACH_D

    def test_a_declared_reach_is_taken(self, tmp_path: Path) -> None:
        upgrades = UPGRADES.replace("    expected_uplift: 0.01\n", "    wake_reach_d: 0\n")
        assert load_upgrades(tmp_path, upgrades=upgrades).spec.wake_reach_d == 0.0

    def test_both_is_rejected(self, tmp_path: Path) -> None:
        upgrades = UPGRADES.replace("    expected_uplift: 0.01\n", "    expected_uplift: 0.01\n    wake_reach_d: 3\n")
        with pytest.raises(ValueError, match="at most one"):
            load_upgrades(tmp_path, upgrades=upgrades)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (HEAD + UPGRADES, "lists its analyses"),
        (HEAD + ANALYSES, "has no upgrades"),
        (HEAD + "upgrades: []\n" + ANALYSES, "at least one upgrade"),
        (HEAD + UPGRADES + ANALYSES + "timing:\n  mode: prepost\n", r"not with \['timing'\]"),
        (HEAD.replace("  excluded: [T05]", "  upgraded: [T01]") + UPGRADES + ANALYSES, "turbines.upgraded"),
        (HEAD.replace("turbines.csv\n", "turbines.csv\n  works: aeroup.csv\n") + UPGRADES + ANALYSES, "data.works"),
        (HEAD + UPGRADES.replace("name: AeroUp", "name: TuneUp") + ANALYSES, "own name"),
        (HEAD + UPGRADES + ANALYSES.replace("upgrade: TuneUp", "upgrade: Mystery"), "Mystery"),
        (HEAD + UPGRADES + ANALYSES + "  - {name: TuneUp, upgrade: AeroUp}\n", "own name"),
        (HEAD + UPGRADES + ANALYSES.replace("upgrade: TuneUp", "upgrade: TuneUp\n    subjects: [T03]"), "did not"),
        (HEAD + UPGRADES + ANALYSES.replace("upgrade: TuneUp", "upgrade: TuneUp\n    colour: red"), "unknown keys"),
        (HEAD + UPGRADES + ANALYSES.replace("upgrade: TuneUp", "upgrade: TuneUp\n    report: [p50]"), "p50"),
        (HEAD + UPGRADES + ANALYSES.replace("upgrade: TuneUp", "upgrade: TuneUp\n    report: []"), "nothing"),
        (HEAD + UPGRADES + ANALYSES.replace("name: TuneUp", "name: a/b"), "path separator"),
        (HEAD + UPGRADES.replace("name: AeroUp", "name: AeroUp\n    colour: red") + ANALYSES, "unknown keys"),
        (HEAD + UPGRADES.replace("turbines: [T01, T02]", "turbines: [T01]") + ANALYSES, "not among its turbines"),
        (HEAD + UPGRADES.replace("turbines: [T01, T02]", "turbines: [T01, T09]") + ANALYSES, "T09"),
        (
            HEAD + UPGRADES.replace("    installation_windows: aeroup.csv\n", "    turbines: [T03]\n") + ANALYSES,
            "needs a timing or installation windows",
        ),
        (
            HEAD
            + UPGRADES.replace("    installation_windows: aeroup.csv\n", "    turbines: [T03, T04]\n").replace(
                "  - name: TuneUp",
                "  - name: Other\n    installation_windows:\n"
                "      - {turbine: T03, start: 2017-01-01T00:00:00Z, end: 2017-01-02T00:00:00Z}\n  - name: TuneUp",
            )
            + ANALYSES,
            "needs a timing or installation windows",
        ),
        (HEAD + "upgrades:\n  - turbines: [T01]\n" + ANALYSES, "with a name"),
        (HEAD + UPGRADES + "analyses:\n  - {name: x}\n", "the upgrade it measures"),
        (HEAD + UPGRADES + "analyses:\n  - {name: x, upgrade: TuneUp, subjects: T01}\n", "list of names"),
    ],
)
def test_bad_declarations_raise(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        load(tmp_path, text)


def test_an_upgrade_without_timing_needs_a_window_for_each_turbine(tmp_path: Path) -> None:
    upgrades = UPGRADES.replace(
        "    installation_windows: aeroup.csv\n",
        "    installation_windows: aeroup.csv\n    turbines: [T01, T02, T03, T04, T05]\n",
    )
    with pytest.raises(ValueError, match=r"\['T05'\] have none"):
        load_upgrades(tmp_path, upgrades=upgrades)


class TestInlineInstallationWindows:
    @pytest.mark.parametrize(
        ("window", "match"),
        [
            ("{turbine: ALL, start: 2018-01-01T00:00:00Z, end: 2018-01-02T00:00:00Z}", "one turbine"),
            ("{turbine: T01, start: 2018-01-01T00:00:00Z}", "exactly the keys"),
            ("{turbine: T01, start: 2018-01-02T00:00:00Z, end: 2018-01-01T00:00:00Z}", "not after its start"),
        ],
    )
    def test_a_bad_window_raises(self, tmp_path: Path, window: str, match: str) -> None:
        upgrades = UPGRADES.replace(
            "      - {turbine: T02, start: 2018-03-14T07:00:00Z, end: 2018-03-14T18:00:00Z}", f"      - {window}"
        )
        with pytest.raises(ValueError, match=match):
            load_upgrades(tmp_path, upgrades=upgrades)

    def test_windows_that_are_not_a_list_or_file_raise(self, tmp_path: Path) -> None:
        upgrades = UPGRADES.replace("installation_windows: aeroup.csv", "installation_windows: {T01: 3}")
        with pytest.raises(ValueError, match="table file or a list"):
            load_upgrades(tmp_path, upgrades=upgrades)


class TestTheComposedRunsCampaign:
    def test_a_subject_with_its_changeover_among_other_windows_needs_a_shared_changeover(self, tmp_path: Path) -> None:
        upgrades = UPGRADES.replace("    timing: {mode: prepost, changeover: 2018-03-14T18:00:00Z}\n", "")
        with pytest.raises(ValueError, match="shared changeover"):
            load_upgrades(tmp_path, upgrades=upgrades)

    def test_a_window_over_the_changeover_raises(self, tmp_path: Path) -> None:
        upgrades = UPGRADES.replace(
            "end: 2018-03-14T18:00:00Z}\n      - {turbine: T02", "end: 2018-03-15T00:00:00Z}\n      - {turbine: T02"
        )
        with pytest.raises(ValueError, match=r"over timing\.changeover"):
            load_upgrades(tmp_path, upgrades=upgrades)

    def test_an_analysis_naming_labels_has_none(self, tmp_path: Path) -> None:
        analyses = ANALYSES + "  - {name: sector, upgrade: TuneUp, include_labels: [in sector]}\n"
        declaration = load_upgrades(
            tmp_path, analyses=analyses, extra="labels:\n  - {name: in sector, column: sector}\n"
        )
        assert set(declaration.specs) == {"TuneUp"}
        (_, sector) = declaration.analyses
        assert sector.include_labels == ("in sector",)
        with pytest.raises(ValueError, match="the composed run does not read"):
            declaration.spec_for("sector")
        assert declaration.resolved()["analyses"][1]["subjects"] is None

    def test_the_bridge_rejects_labels_and_unknown_upgrades(self) -> None:
        upgrade = DeclaredUpgrade(
            name="u", turbines=("T01",), timing=utc("2018-01-01"), installation_windows={}, wake_reach_d=1.0
        )
        kwargs = {
            "upgrades": [upgrade],
            "layout": None,
            "excluded": [],
            "rated_power_kw": 1.0,
            "turbine_col": "t",
            "north_offsets": None,
        }
        with pytest.raises(ValueError, match="does not read labels"):
            campaign_spec(DeclaredAnalysis(name="a", upgrade="u", exclude_labels=("x",)), **kwargs)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="the upgrades are"):
            campaign_spec(DeclaredAnalysis(name="a", upgrade="v"), **kwargs)  # type: ignore[arg-type]


class TestLabels:
    def test_labels_are_read(self, tmp_path: Path) -> None:
        extra = """
labels:
  - {name: curtailed, column: curtailed, withholds: uplift}
  - {name: in sector, column: sector, withholds: [uplift, northing]}
  - {name: note, column: note}
"""
        curtailed, sector, note = load_upgrades(tmp_path, extra=extra).labels
        assert curtailed.withholds == {UPLIFT}
        assert sector.withholds == {UPLIFT, NORTHING}
        assert note.column == "note"
        assert not note.withholds
        assert load_upgrades(tmp_path, extra=extra).resolved()["labels"][1] == {
            "name": "in sector",
            "column": "sector",
            "withholds": ["northing", "uplift"],
        }

    @pytest.mark.parametrize(
        ("labels", "match"),
        [
            ("labels: {name: x}\n", "list of"),
            ("labels:\n  - {name: x}\n", "a name and a column"),
            ("labels:\n  - {name: x, column: x, colour: red}\n", "unknown keys"),
            ("labels:\n  - {name: x, column: x}\n  - {name: x, column: y}\n", "own name"),
            ("labels:\n  - {name: x, column: x, withholds: [power]}\n", "may withhold only"),
            ("labels:\n  - {name: x, column: x}\n", "the declared labels are"),
        ],
    )
    def test_bad_labels_raise(self, tmp_path: Path, labels: str, match: str) -> None:
        analyses = ANALYSES + ("  - {name: s, upgrade: TuneUp, exclude_labels: [y]}\n" if "declared" in match else "")
        with pytest.raises(ValueError, match=match):
            load_upgrades(tmp_path, analyses=analyses, extra=labels)

    def test_a_single_upgrade_declaration_takes_labels(self, tmp_path: Path) -> None:
        labels = load(tmp_path, SHORTHAND + "labels:\n  - {name: x, column: x, withholds: uplift}\n").labels
        assert [label.name for label in labels] == ["x"]


class TestNonTurbineMeasurements:
    MAST = """
non_turbine_measurements:
  - name: mast
    latitude: 57.45
    longitude: -3.28
    records: mast.parquet
    columns: {wind_speed: ws}
"""

    def test_a_mast_is_read_and_its_records_loaded(self, tmp_path: Path) -> None:
        index = pd.date_range("2018-01-01", periods=3, freq="10min", tz="UTC")
        pd.DataFrame({"ws": [5.0, 6.0, 7.0]}, index=index).to_parquet(tmp_path / "mast.parquet")
        (mast,) = load_upgrades(tmp_path, extra=self.MAST).non_turbine_measurements
        assert (mast.name, mast.latitude, mast.longitude) == ("mast", 57.45, -3.28)
        assert mast.columns == {"wind_speed": "ws"}
        loaded = mast.load()
        assert loaded.records["ws"].tolist() == [5.0, 6.0, 7.0]
        assert loaded.columns == {"wind_speed": "ws"}
        assert load_upgrades(tmp_path, extra=self.MAST).resolved()["non_turbine_measurements"][0]["name"] == "mast"

    def test_records_without_a_time_index_raise_when_loaded(self, tmp_path: Path) -> None:
        pd.DataFrame({"ws": [5.0]}).to_parquet(tmp_path / "mast.parquet")
        (mast,) = load_upgrades(tmp_path, extra=self.MAST).non_turbine_measurements
        with pytest.raises(TypeError, match="time index"):
            mast.load()

    @pytest.mark.parametrize(
        ("change", "match"),
        [
            (("name: mast", "name: T01"), "turbine's name"),
            (("records: mast.parquet", "records: absent.parquet"), "absent.parquet"),
            (("    columns: {wind_speed: ws}\n", "    colour: red\n"), "unknown keys"),
            (("    columns: {wind_speed: ws}", "    columns: [ws]"), "map a role"),
            (("    latitude: 57.45\n", ""), "latitude, longitude and records"),
            (("latitude: 57.45", "latitude: north"), "finite number"),
        ],
    )
    def test_bad_measurements_raise(self, tmp_path: Path, change: tuple[str, str], match: str) -> None:
        (tmp_path / "mast.parquet").write_bytes(b"")
        with pytest.raises((ValueError, FileNotFoundError), match=match):
            load_upgrades(tmp_path, extra=self.MAST.replace(*change))

    def test_two_of_one_name_raise(self, tmp_path: Path) -> None:
        (tmp_path / "mast.parquet").write_bytes(b"")
        twice = self.MAST + self.MAST.replace("non_turbine_measurements:\n", "")
        with pytest.raises(ValueError, match="own name"):
            load_upgrades(tmp_path, extra=twice)

    def test_they_are_a_list(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="is a list"):
            load_upgrades(tmp_path, extra="non_turbine_measurements: {name: mast}\n")


class TestTheSingleUpgradeKeys:
    def test_they_declare_one_upgrade_and_one_analysis(self, tmp_path: Path) -> None:
        declaration = load(tmp_path, SHORTHAND)
        (upgrade,) = declaration.upgrades
        (analysis,) = declaration.analyses
        assert upgrade.name == SHORTHAND_UPGRADE
        assert upgrade.turbines == ("T01", "T02")
        assert analysis.name == "demo"
        assert analysis.upgrade == SHORTHAND_UPGRADE
        assert declaration.shorthand

    def test_analyses_without_upgrades_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="has no upgrades"):
            load(tmp_path, SHORTHAND + ANALYSES)


class TestRenamedKeys:
    def test_the_old_and_new_names_read_the_same(self, tmp_path: Path) -> None:
        old = load(tmp_path, SHORTHAND.replace("installation_windows: shorthand.csv", "works: shorthand.csv")).spec
        assert old.works == load(tmp_path, SHORTHAND).spec.works

    def test_both_names_at_once_raise(self, tmp_path: Path) -> None:
        both = SHORTHAND.replace(
            "  installation_windows: shorthand.csv", "  installation_windows: shorthand.csv\n  works: shorthand.csv"
        )
        with pytest.raises(ValueError, match="old name works"):
            load(tmp_path, both)

    def test_operating_state_overrides_take_either_name(self, tmp_path: Path) -> None:
        entry = "  - {turbine: T03, start: 2017-08-01T00:00:00Z, end: 2017-08-02T00:00:00Z, state: partial downtime}\n"
        new = load(tmp_path, SHORTHAND + "operating_state_overrides:\n" + entry).spec
        old = load(tmp_path, SHORTHAND + "label_overrides:\n" + entry).spec
        assert new.operating_state_overrides == old.operating_state_overrides
        assert len(new.operating_state_overrides) == 1

    def test_the_site_state_keys_take_either_name(self, tmp_path: Path) -> None:
        new = """
operating_state:
  site_state_column: state
  site_states:
    curtailment: {northing: valid, waking: part waking, uplift: not valid}
"""
        old = new.replace("site_state_column", "label_column").replace("site_states", "labels")
        assert load(tmp_path, SHORTHAND + new).operating_state == load(tmp_path, SHORTHAND + old).operating_state
        assert load(tmp_path, SHORTHAND + new).operating_state.site_state_column == "state"
