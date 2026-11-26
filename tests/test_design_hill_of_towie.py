"""Tests for the Hill of Towie campaign design example."""

from pathlib import Path

from examples.v1.design_hill_of_towie import DOCS_MAPS, main


def test_the_example_designs_hill_of_towie_and_writes_its_maps(tmp_path: Path) -> None:
    designs = main(out_dir=tmp_path / "design", docs_images_dir=tmp_path / "images")
    assert sorted(designs) == ["0d", "12d", "20d"]
    for reach, design in designs.items():
        assert design.compliance.compliant
        assert len(design.test_turbines) == design.max_test_turbines
        assert "T17" not in design.test_turbines
        assert (tmp_path / "design" / f"reach_{reach}" / "compliance.csv").is_file()
    assert [designs[r].wake_reach_d for r in ("0d", "12d", "20d")] == [0.0, 12.0, 20.0]
    assert len(designs["0d"].test_turbines) > len(designs["12d"].test_turbines) > len(designs["20d"].test_turbines)
    assert sorted(p.name for p in (tmp_path / "images").iterdir()) == sorted(DOCS_MAPS)


def test_the_example_can_leave_the_docs_alone(tmp_path: Path) -> None:
    main(out_dir=tmp_path / "design", docs_images_dir=None)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["design"]
