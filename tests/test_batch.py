"""Tests for the batch layer.

The contract worth defending is the array shape: same length, same
order, one bad design isolated to its own slot. Most tests stub the
upstream model so they exercise that contract without paying for a
PyBaMM solve; ``test_real_model_*`` runs the genuine thing on the
example designs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cell_performance_batch import run_batch
from cell_performance_batch.batch import COMPARISON_KEYS, RESULT_BYTE_BUDGET

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = json.loads((REPO_ROOT / "examples" / "two_designs.json").read_text())


@pytest.fixture()
def example() -> dict:
    return copy.deepcopy(EXAMPLE)


def _stub_model(monkeypatch, fake):
    """Replace the upstream entry point the batch imports lazily."""
    monkeypatch.setattr(
        "cell_performance_batch.cell_performance.calculate_cell_performance",
        fake,
        raising=True,
    )


class TestEnvelope:
    def test_empty_batch_is_valid_and_cheap(self):
        # This is the exact payload the Protos build pipeline sends as
        # its dry-run sample, so it must succeed without a model run.
        result = run_batch({"designs": [], "simulation_parameters": {}})
        assert result["results"] == []
        assert result["summary"]["design_count"] == 0
        assert result["summary"]["succeeded"] == 0

    def test_design_without_protocol_and_no_batch_default_is_rejected(self, example):
        example.pop("simulation_parameters")
        with pytest.raises(ValidationError, match="simulation_parameters"):
            run_batch(example)

    def test_unknown_top_level_key_is_rejected(self, example):
        example["not_a_field"] = 1
        with pytest.raises(ValidationError):
            run_batch(example)

    def test_per_design_protocol_replaces_batch_level(self, monkeypatch, example):
        seen: list[dict] = []

        def fake(payload, progress_callback=None):
            seen.append(payload["simulation_parameters"])
            return {"cell_nominal_capacity_Ah": 1.0}

        _stub_model(monkeypatch, fake)
        example["designs"][1]["simulation_parameters"] = {"start_soc_pct": 42.0}
        run_batch(example)

        assert seen[0] == example["simulation_parameters"]
        # Replaced whole, not merged — no batch-level keys leak in.
        assert seen[1] == {"start_soc_pct": 42.0}


class TestFailureIsolation:
    def test_one_failing_design_does_not_kill_the_batch(self, monkeypatch, example):
        def fake(payload, progress_callback=None):
            loading = payload["cell_parameters"][
                "positive_electrode_mass_loading_mg_cm2"
            ]
            if loading > 20:
                raise RuntimeError("solver diverged")
            return {"cell_nominal_capacity_Ah": 67.3}

        _stub_model(monkeypatch, fake)
        result = run_batch(example)

        assert [r["index"] for r in result["results"]] == [0, 1]
        assert result["results"][0]["status"] == "ok"
        assert result["results"][1]["status"] == "error"
        assert "solver diverged" in result["results"][1]["error"]
        assert result["results"][1]["kpis"] is None
        assert (result["summary"]["succeeded"], result["summary"]["failed"]) == (1, 1)

    def test_empty_upstream_result_counts_as_an_error(self, monkeypatch, example):
        # calculate_cell_performance returns {} instead of raising when
        # its internal computation returns None.
        _stub_model(monkeypatch, lambda payload, progress_callback=None: {})
        result = run_batch(example)

        assert [r["status"] for r in result["results"]] == ["error", "error"]
        assert "no KPIs" in result["results"][0]["error"]

    def test_fail_fast_skips_the_remainder_but_keeps_the_slots(
        self, monkeypatch, example
    ):
        def fake(payload, progress_callback=None):
            raise ValueError("bad design")

        _stub_model(monkeypatch, fake)
        example["fail_fast"] = True
        result = run_batch(example)

        assert len(result["results"]) == len(example["designs"])
        assert result["results"][0]["status"] == "error"
        assert result["results"][1]["status"] == "skipped"
        assert result["summary"]["skipped"] == 1

    def test_identity_fields_survive_a_failure(self, monkeypatch, example):
        _stub_model(
            monkeypatch,
            lambda payload, progress_callback=None: (_ for _ in ()).throw(
                RuntimeError("boom")
            ),
        )
        result = run_batch(example)

        assert result["results"][0]["id"] == "pouch-80ah-baseline"
        assert result["results"][0]["name"] == "Baseline (15 mg/cm2 cathode)"


class TestResultDetail:
    def _fake_kpis(self) -> dict:
        return {
            "cell_nominal_capacity_Ah": 67.3,
            "cell_mass_g": 1111.0,
            "electrode_porosity_positive": 0.31,
            "c3_discharge_timeseries": {"time_s": [0.0, 1.0]},
            "experiment_results": {"cycle": {}},
            "bill_of_materials": {"items": []},
        }

    def test_full_keeps_everything(self, monkeypatch, example):
        _stub_model(monkeypatch, lambda p, progress_callback=None: self._fake_kpis())
        example["result_detail"] = "full"
        kpis = run_batch(example)["results"][0]["kpis"]
        assert "c3_discharge_timeseries" in kpis
        assert "bill_of_materials" in kpis

    def test_kpis_drops_only_the_bulk_payloads(self, monkeypatch, example):
        _stub_model(monkeypatch, lambda p, progress_callback=None: self._fake_kpis())
        example["result_detail"] = "kpis"
        kpis = run_batch(example)["results"][0]["kpis"]
        assert "c3_discharge_timeseries" not in kpis
        assert "experiment_results" not in kpis
        assert "bill_of_materials" not in kpis
        assert kpis["electrode_porosity_positive"] == 0.31

    def test_summary_keeps_only_comparison_scalars(self, monkeypatch, example):
        _stub_model(monkeypatch, lambda p, progress_callback=None: self._fake_kpis())
        example["result_detail"] = "summary"
        kpis = run_batch(example)["results"][0]["kpis"]
        assert set(kpis) == set(COMPARISON_KEYS)
        assert kpis["cell_nominal_capacity_Ah"] == 67.3


class TestResultBudget:
    """The result travels as one line of container stdout.

    containerd splits log lines at 16 KiB, and the platform's reader
    scans line by line for one that parses — so an oversized result
    fails the run with "Could not find JSON result in pod output" even
    though the container exited 0 and every design succeeded.
    """

    def _bulky(self, samples: int):
        def fake(payload, progress_callback=None):
            return {
                "cell_nominal_capacity_Ah": 67.3,
                "cell_mass_g": 1111.0,
                "c3_discharge_timeseries": {
                    "time_s": [float(i) for i in range(samples)],
                    "voltage_V": [3.7] * samples,
                },
            }

        return fake

    def test_oversized_result_is_degraded_not_dropped(self, monkeypatch, example):
        _stub_model(monkeypatch, self._bulky(2000))
        example["result_detail"] = "full"
        result = run_batch(example)

        assert len(json.dumps(result)) <= example.get(
            "max_result_bytes", RESULT_BYTE_BUDGET
        )
        # Degraded, not emptied: the KPIs a sweep is for are still here.
        assert result["summary"]["succeeded"] == 2
        assert result["results"][0]["kpis"]["cell_nominal_capacity_Ah"] == 67.3
        assert result["summary"]["comparison"][1]["cell_mass_g"] == 1111.0

    def test_truncation_is_declared_with_what_was_dropped(self, monkeypatch, example):
        _stub_model(monkeypatch, self._bulky(2000))
        example["result_detail"] = "full"
        truncated = run_batch(example)["truncated"]

        assert truncated["requested_result_detail"] == "full"
        assert "kpis" in truncated["applied"]
        assert truncated["original_size_bytes"] > truncated["budget_bytes"]
        assert truncated["size_bytes"] <= truncated["budget_bytes"]
        assert truncated["reason"] and truncated["hint"]

    def test_a_result_that_already_fits_is_untouched(self, monkeypatch, example):
        _stub_model(monkeypatch, lambda p, progress_callback=None: {"cell_mass_g": 1.0})
        result = run_batch(example)

        assert "truncated" not in result
        assert result["summary"]["result_detail"] == "kpis"

    def test_budget_zero_opts_out_entirely(self, monkeypatch, example):
        _stub_model(monkeypatch, self._bulky(2000))
        example["result_detail"] = "full"
        example["max_result_bytes"] = 0
        result = run_batch(example)

        assert "truncated" not in result
        assert len(json.dumps(result)) > RESULT_BYTE_BUDGET
        assert "c3_discharge_timeseries" in result["results"][0]["kpis"]

    def test_the_last_rungs_keep_the_run_reportable(self, monkeypatch, example):
        # Budget too small for any per-design payload: the ladder must
        # still return a parseable envelope with the run's outcome.
        _stub_model(monkeypatch, self._bulky(2000))
        example["result_detail"] = "full"
        example["max_result_bytes"] = 600
        result = run_batch(example)

        assert result["summary"]["design_count"] == 2
        assert result["summary"]["succeeded"] == 2
        assert [r["status"] for r in result["results"]] == ["ok", "ok"]
        assert all(r["kpis"] is None for r in result["results"])
        assert "truncated" in result

    @pytest.mark.slow()
    def test_the_real_two_design_sweep_fits_by_default(self, example):
        # The regression that produced the platform error: the shipped
        # default must fit inside one log line for the bundled example.
        example.pop("result_detail", None)
        rendered = json.dumps(run_batch(example))
        assert len(rendered) < 16 * 1024


class TestComparisonTable:
    def test_one_row_per_design_including_failures(self, monkeypatch, example):
        def fake(payload, progress_callback=None):
            if payload["cell_parameters"]["positive_coating_thickness_um"] > 90:
                raise RuntimeError("nope")
            return {"cell_nominal_capacity_Ah": 67.3, "cell_mass_g": 1111.0}

        _stub_model(monkeypatch, fake)
        comparison = run_batch(example)["summary"]["comparison"]

        assert [row["index"] for row in comparison] == [0, 1]
        assert comparison[0]["cell_nominal_capacity_Ah"] == 67.3
        assert comparison[1]["status"] == "error"
        # Failed rows still carry every column, valued null.
        assert comparison[1]["cell_nominal_capacity_Ah"] is None


class TestProgress:
    def test_progress_reaches_100_and_is_monotonic(self, monkeypatch, example):
        _stub_model(monkeypatch, lambda p, progress_callback=None: {"cell_mass_g": 1.0})
        seen: list[int] = []
        run_batch(example, progress_callback=lambda pct, msg: seen.append(pct))

        assert seen == sorted(seen)
        assert seen[-1] == 100


@pytest.mark.slow()
class TestRealModel:
    """End-to-end against the vendored PyBaMM pipeline."""

    def test_two_designs_produce_distinct_kpis(self, example):
        result = run_batch(example)

        assert result["summary"]["succeeded"] == 2
        first, second = (
            row["cell_nominal_capacity_Ah"] for row in result["summary"]["comparison"]
        )
        assert first is not None and second is not None
        # Higher cathode loading must yield more capacity — if this
        # flips, the vendored model or its inputs drifted.
        assert second > first

    def test_parallel_path_matches_the_sequential_one(self, example):
        # Stubbing can't cross a process boundary, so this is the only
        # way to cover the pool: run both for real and compare.
        sequential = run_batch(copy.deepcopy(example))
        example["max_workers"] = 2
        parallel = run_batch(example)

        assert [r["index"] for r in parallel["results"]] == [0, 1]
        for seq, par in zip(
            sequential["summary"]["comparison"],
            parallel["summary"]["comparison"],
            strict=True,
        ):
            for key in COMPARISON_KEYS:
                assert seq[key] == par[key]

    def test_invalid_cell_parameters_land_as_one_error_entry(self, example):
        example["designs"][1]["cell_parameters"][
            "positive_electrode_mass_loading_mg_cm2"
        ] = -5.0
        result = run_batch(example)

        assert result["results"][0]["status"] == "ok"
        assert result["results"][1]["status"] == "error"
