from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from pandas import (  # pyright: ignore[reportMissingImports]
    DataFrame,
    DatetimeIndex,
    MultiIndex,
    Timestamp,
    date_range,
)
from pandas.testing import assert_frame_equal  # pyright: ignore[reportMissingImports]
from swmm.pandas import example_out_path  # pyright: ignore[reportMissingImports]

from swmm_bench.comparator import (
    _MAX_GRAPH_POINTS_PER_SERIES,
    _cell_distance,
    _compare_output_frames,
    _sample_output_positions,
    compare_all_outputs,
    compare_outs,
)
from swmm_bench.models import EngineResult, OUTPUT_DISTANCE_METRIC_NRMSE
from swmm_bench.output import extract_output_frame, extract_output_series


class OutputExtractionTests(unittest.TestCase):
    def test_package_fixture_extracts_a_semantic_wide_frame(self) -> None:
        preloaded = extract_output_frame(example_out_path)
        streamed = extract_output_frame(example_out_path, preload=False)

        self.assertIsInstance(preloaded.index, DatetimeIndex)
        self.assertEqual(preloaded.index.name, "datetime")
        self.assertIsInstance(preloaded.columns, MultiIndex)
        self.assertEqual(
            preloaded.columns.names,
            ["element_type", "element_name", "attribute"],
        )
        self.assertEqual(
            set(preloaded.columns.get_level_values("element_type")),
            {"subcatchment", "node", "link", "system"},
        )
        self.assertIn(("node", "JUNC1", "hydraulic_head"), preloaded.columns)
        self.assertIn(("system", "system", "outfall_flows"), preloaded.columns)
        assert_frame_equal(preloaded, streamed, check_dtype=False)

    def test_output_series_preserve_each_semantic_column(self) -> None:
        frame = extract_output_frame(example_out_path)
        series = extract_output_series(example_out_path)

        self.assertEqual(len(series), len(frame.columns))
        self.assertTrue(
            all(
                table.columns.equals(frame.loc[:, [column]].columns)
                for column, table in zip(frame.columns, series.values())
            )
        )

    def test_same_output_has_zero_distance(self) -> None:
        comparison = compare_outs(
            example_out_path,
            example_out_path,
            "a",
            "b",
            "model.inp",
            "model.inp",
        )

        self.assertEqual(comparison.metric, OUTPUT_DISTANCE_METRIC_NRMSE)
        self.assertEqual(comparison.overall_distance, 0.0)
        self.assertEqual(comparison.section_comparisons, [])
        self.assertEqual(comparison.graphical_series, [])
        self.assertFalse(comparison.details_retained)

    def test_graphical_series_can_be_forced_at_zero_distance(self) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        frame = DataFrame(
            [[1.0], [2.0]],
            index=date_range("2026-08-03", periods=2, freq="5min"),
            columns=columns,
        )

        comparison = _compare_output_frames(
            frame,
            frame,
            "a",
            "b",
            "model.inp",
            "model.inp",
            retain_tabular=False,
            include_all_comparisons=True,
        )

        self.assertEqual(comparison.overall_distance, 0.0)
        self.assertEqual(len(comparison.graphical_series), 1)
        self.assertTrue(comparison.details_retained)

    def test_graphical_series_aligns_semantic_values_and_missing_timestamps(
        self,
    ) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "hydraulic_head")],
            names=["element_type", "element_name", "attribute"],
        )
        frame_a = DataFrame(
            [[1.0], [2.0]],
            index=DatetimeIndex(
                [Timestamp("2020-01-01"), Timestamp("2020-01-01 00:05")]
            ),
            columns=columns,
        )
        frame_b = DataFrame(
            [[2.5], [3.0]],
            index=DatetimeIndex(
                [Timestamp("2020-01-01 00:05"), Timestamp("2020-01-01 00:10")]
            ),
            columns=columns,
        )

        comparison = _compare_output_frames(
            frame_a,
            frame_b,
            "a",
            "b",
            "model.inp",
            "model.inp",
        )

        series = comparison.graphical_series[0]
        self.assertEqual(
            (series.element_type, series.element_name, series.attribute),
            ("node", "J1", "hydraulic_head"),
        )
        self.assertEqual(series.values_a, [1.0, 2.0])
        self.assertEqual(series.values_b, [None, 2.5])
        section = comparison.section_comparisons[0]
        self.assertIsNotNone(section.numeric_distance)
        self.assertIsNotNone(section.missing_fraction)
        self.assertAlmostEqual(cast(float, section.numeric_distance), 0.2)
        self.assertAlmostEqual(cast(float, section.missing_fraction), 0.5)
        self.assertAlmostEqual(section.distance, 0.5 + 0.5 * 0.2)
        self.assertEqual(section.finite_pair_count, 1)
        self.assertEqual(section.missing_count, 1)
        self.assertEqual(comparison.timeline_coverage.trailing_timestamp_count_b, 1)
        self.assertEqual(series.distance, section.distance)
        self.assertFalse(series.sampled)

    def test_trailing_periods_are_coverage_not_value_distance(self) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        shared_index = date_range("2020-01-01", periods=3, freq="5min")
        extended_index = date_range("2020-01-01", periods=6, freq="5min")
        frame_a = DataFrame([1.0, 2.0, 3.0], index=shared_index, columns=columns)
        frame_b = DataFrame(
            [1.0, 2.0, 3.0, 40.0, 50.0, 60.0],
            index=extended_index,
            columns=columns,
        )

        forward = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        reverse = _compare_output_frames(frame_b, frame_a, "b", "a", "m", "m")
        frame_b_with_shared_difference = frame_b.copy()
        frame_b_with_shared_difference.iloc[1, 0] = 20.0
        differing = _compare_output_frames(
            frame_a, frame_b_with_shared_difference, "a", "b", "m", "m"
        )

        self.assertEqual(forward.metric, "normalized-rmse-shared-timeline-v2")
        self.assertEqual(forward.overall_distance, 0.0)
        self.assertEqual(reverse.overall_distance, 0.0)
        self.assertEqual(forward.section_comparisons[0].missing_fraction, 0.0)
        self.assertEqual(forward.timeline_coverage.timestamp_count_a, 3)
        self.assertEqual(forward.timeline_coverage.timestamp_count_b, 6)
        self.assertEqual(forward.timeline_coverage.shared_timestamp_count, 3)
        self.assertEqual(forward.timeline_coverage.trailing_timestamp_count_a, 0)
        self.assertEqual(forward.timeline_coverage.trailing_timestamp_count_b, 3)
        self.assertEqual(reverse.timeline_coverage.trailing_timestamp_count_a, 3)
        self.assertEqual(reverse.timeline_coverage.trailing_timestamp_count_b, 0)
        self.assertGreater(differing.overall_distance, 0.0)

    def test_internal_gap_is_penalized_while_trailing_periods_are_coverage(
        self,
    ) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        frame_a = DataFrame(
            [1.0, 3.0, 4.0],
            index=DatetimeIndex(
                [
                    Timestamp("2020-01-01 00:00"),
                    Timestamp("2020-01-01 00:10"),
                    Timestamp("2020-01-01 00:15"),
                ]
            ),
            columns=columns,
        )
        frame_b = DataFrame(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            index=date_range("2020-01-01", periods=5, freq="5min"),
            columns=columns,
        )

        forward = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        reverse = _compare_output_frames(frame_b, frame_a, "b", "a", "m", "m")

        self.assertEqual(forward.overall_distance, 0.25)
        self.assertEqual(reverse.overall_distance, 0.25)
        section = forward.section_comparisons[0]
        self.assertEqual(section.missing_count, 1)
        self.assertEqual(section.timestamp_count, 4)
        self.assertEqual(forward.timeline_coverage.trailing_timestamp_count_b, 1)
        self.assertEqual(reverse.timeline_coverage.trailing_timestamp_count_a, 1)

    def test_output_distance_uses_symmetric_rms_normalized_rmse(self) -> None:
        columns = MultiIndex.from_tuples(
            [("link", "C1", "flow")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=3, freq="min")
        frame_a = DataFrame([1.0, 2.0, 3.0], index=index, columns=columns)
        frame_b = DataFrame([1.0, 2.0, 4.0], index=index, columns=columns)

        forward = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        reverse = _compare_output_frames(frame_b, frame_a, "b", "a", "m", "m")
        scaled = _compare_output_frames(
            frame_a * 1000.0,
            frame_b * 1000.0,
            "a",
            "b",
            "m",
            "m",
        )

        expected = 1.0 / math.sqrt(21.0)
        self.assertAlmostEqual(forward.overall_distance, expected)
        self.assertAlmostEqual(reverse.overall_distance, expected)
        self.assertAlmostEqual(scaled.overall_distance, expected)
        numeric_distance = forward.section_comparisons[0].numeric_distance
        self.assertIsNotNone(numeric_distance)
        self.assertAlmostEqual(cast(float, numeric_distance), expected)

    def test_event_scale_prevents_near_zero_noise_from_dominating(self) -> None:
        columns = MultiIndex.from_tuples(
            [("link", "C1", "flow_rate")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("1977-07-15", periods=8, freq="10min")
        values_a = [
            0.0,
            -237.9263,
            -415.4236,
            27.6930,
            0.03813,
            0.02513,
            -0.03188,
            0.00914,
        ]
        values_b = [
            0.0,
            -236.7247,
            -417.1129,
            27.7395,
            0.00182,
            -0.05181,
            0.00210,
            0.01837,
        ]
        frame_a = DataFrame(values_a, index=index, columns=columns)
        frame_b = DataFrame(values_b, index=index, columns=columns)

        comparison = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        legacy_distance = sum(
            _cell_distance(value_a, value_b)[0]
            for value_a, value_b in zip(values_a, values_b)
        ) / len(values_a)

        self.assertGreater(legacy_distance, 0.4)
        self.assertLess(comparison.overall_distance, 0.01)
        self.assertEqual(comparison.section_comparisons[0].missing_fraction, 0.0)

    def test_paired_nulls_are_neutral_and_one_sided_nulls_are_missing(self) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=3, freq="min")
        frame_a = DataFrame([0.0, None, 1.0], index=index, columns=columns)
        frame_b = DataFrame([0.0, None, None], index=index, columns=columns)

        comparison = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        section = comparison.section_comparisons[0]

        self.assertEqual(section.numeric_distance, 0.0)
        self.assertIsNotNone(section.missing_fraction)
        self.assertAlmostEqual(cast(float, section.missing_fraction), 1.0 / 3.0)
        self.assertAlmostEqual(section.distance, 1.0 / 3.0)
        self.assertEqual(section.finite_pair_count, 1)
        self.assertEqual(section.missing_count, 1)
        self.assertEqual(section.both_null_count, 1)

    def test_absent_series_has_full_penalty_and_series_are_equally_weighted(
        self,
    ) -> None:
        columns_a = MultiIndex.from_tuples(
            [("node", "J1", "depth"), ("node", "J1", "volume")],
            names=["element_type", "element_name", "attribute"],
        )
        columns_b = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=4, freq="min")
        frame_a = DataFrame(
            [[1.0, 10.0], [1.0, 11.0], [1.0, 12.0], [1.0, 13.0]],
            index=index,
            columns=columns_a,
        )
        frame_b = DataFrame([1.0, 1.0, 1.0, 1.0], index=index, columns=columns_b)

        comparison = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        distances = {
            section.section_name: section.distance
            for section in comparison.section_comparisons
        }

        self.assertEqual(distances['["node","J1","depth"]'], 0.0)
        self.assertEqual(distances['["node","J1","volume"]'], 1.0)
        self.assertEqual(comparison.overall_distance, 0.5)

    def test_nonfinite_output_value_is_missing_and_scores_finitely(self) -> None:
        columns = MultiIndex.from_tuples(
            [("system", "system", "flow")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=1, freq="min")
        frame_a = DataFrame([math.inf], index=index, columns=columns)
        frame_b = DataFrame([1.0], index=index, columns=columns)

        comparison = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        section = comparison.section_comparisons[0]

        self.assertEqual(section.distance, 1.0)
        self.assertEqual(section.missing_fraction, 1.0)
        self.assertTrue(math.isfinite(comparison.overall_distance))
        self.assertEqual(section.differences[0]["value_a"], "inf")

    def test_output_diagnostics_are_bounded_and_keep_missing_values_first(self) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=150, freq="min")
        frame_a = DataFrame([None] * 10 + [0.0] * 140, index=index, columns=columns)
        frame_b = DataFrame([1.0] * 150, index=index, columns=columns)

        comparison = _compare_output_frames(frame_a, frame_b, "a", "b", "m", "m")
        section = comparison.section_comparisons[0]

        self.assertEqual(section.difference_count, 150)
        self.assertEqual(len(section.differences), 100)
        self.assertTrue(section.differences_truncated)
        self.assertTrue(
            all(row["issue"] == "null value in A" for row in section.differences[:10])
        )
        self.assertTrue(
            all(
                row["issue"] == "numeric difference" for row in section.differences[10:]
            )
        )

    def test_output_comparison_does_not_use_report_table_metric(self) -> None:
        columns = MultiIndex.from_tuples(
            [("link", "C1", "flow")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=2, freq="min")
        frame = DataFrame([1.0, 2.0], index=index, columns=columns)

        with patch("swmm_bench.comparator._compare_tables", side_effect=AssertionError):
            comparison = _compare_output_frames(frame, frame.copy(), "a", "b", "m", "m")

        self.assertEqual(comparison.overall_distance, 0.0)

    def test_sampling_is_deterministic_and_preserves_important_points(self) -> None:
        values_a = [0.0] * 1000
        values_b = [0.0] * 1000
        values_a[250] = 10.0
        values_a[700] = -5.0
        values_b[500] = 20.0

        first = _sample_output_positions(values_a, values_b, 20)
        second = _sample_output_positions(values_a, values_b, 20)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertTrue({0, 250, 500, 700, 999}.issubset(first))

    def test_sampling_does_not_change_full_resolution_distance(self) -> None:
        columns = MultiIndex.from_tuples(
            [("link", "C1", "flow")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=600, freq="min")
        values_a = [0.0] * 600
        values_b = [0.0] * 600
        values_b[300] = 1.0
        frame_a = DataFrame(values_a, index=index, columns=columns)
        frame_b = DataFrame(values_b, index=index, columns=columns)

        comparison = _compare_output_frames(
            frame_a,
            frame_b,
            "a",
            "b",
            "model.inp",
            "model.inp",
        )

        self.assertEqual(comparison.overall_distance, 1.0)
        self.assertEqual(comparison.graphical_series[0].distance, 1.0)
        self.assertEqual(
            len(comparison.graphical_series[0].timestamps),
            _MAX_GRAPH_POINTS_PER_SERIES,
        )
        self.assertTrue(comparison.graphical_series[0].sampled)

    def test_graphical_payload_is_omitted_when_series_exceed_point_budget(self) -> None:
        columns = MultiIndex.from_tuples(
            [
                ("node", "J1", "depth"),
                ("node", "J1", "flow"),
                ("node", "J1", "volume"),
            ],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=3, freq="min")
        frame = DataFrame(
            [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]],
            index=index,
            columns=columns,
        )
        compared = frame.copy()
        compared.iloc[0, 0] = 10.0
        with (
            patch("swmm_bench.comparator._MAX_GRAPH_POINTS_PER_COMPARISON", 4),
            patch.object(
                DataFrame,
                "reindex",
                side_effect=AssertionError("full frames should not be aligned"),
            ),
        ):
            comparison = _compare_output_frames(
                frame,
                compared,
                "a",
                "b",
                "model.inp",
                "model.inp",
            )

        self.assertEqual(comparison.graphical_series, [])
        self.assertIn("3 output series", comparison.graphical_unavailable_reason or "")
        self.assertEqual(len(comparison.section_comparisons), 3)
        self.assertGreater(comparison.overall_distance, 0.01)

    def test_graph_payload_is_omitted_instead_of_hiding_interior_divergence(
        self,
    ) -> None:
        columns = MultiIndex.from_tuples(
            [("node", "J1", "depth"), ("node", "J2", "depth")],
            names=["element_type", "element_name", "attribute"],
        )
        index = date_range("2020-01-01", periods=3, freq="min")
        frame_a = DataFrame(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            index=index,
            columns=columns,
        )
        frame_b = DataFrame(
            [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
            index=index,
            columns=columns,
        )

        with patch("swmm_bench.comparator._MAX_GRAPH_POINTS_PER_COMPARISON", 4):
            comparison = _compare_output_frames(
                frame_a,
                frame_b,
                "a",
                "b",
                "model.inp",
                "model.inp",
            )

        self.assertEqual(comparison.overall_distance, 1.0)
        self.assertEqual(comparison.graphical_series, [])
        self.assertIn(
            "chart payload budget", comparison.graphical_unavailable_reason or ""
        )

    def test_summary_omits_series_details(self) -> None:
        columns = MultiIndex.from_tuples([("node", "J1", "depth")])
        frame_a = DataFrame([[1.0], [2.0]], columns=columns)
        frame_b = DataFrame([[1.0], [3.0]], columns=columns)

        comparison = _compare_output_frames(
            frame_a,
            frame_b,
            "a",
            "b",
            "model.inp",
            "model.inp",
            retain_tabular=False,
            retain_graphical=False,
        )

        self.assertGreater(comparison.overall_distance, 0.0)
        self.assertEqual(comparison.section_comparisons, [])
        self.assertEqual(comparison.graphical_series, [])
        self.assertFalse(comparison.details_retained)

    def test_output_pairing_uses_available_output_artifacts(self) -> None:
        def result(engine_name: str) -> EngineResult:
            return EngineResult(
                engine_path=f"/{engine_name}",
                engine_name=engine_name,
                inp_path="model.inp",
                inp_name="model.inp",
                duration_s=1.0,
                peak_memory_mb=1.0,
                exit_code=None,
                rpt_path=None,
                stdout="",
                stderr="",
                error=None,
                out_path=str(example_out_path),
            )

        comparisons = compare_all_outputs([result("a"), result("b")])

        self.assertEqual(len(comparisons), 1)
        self.assertEqual(comparisons[0].overall_distance, 0.0)

    def test_output_comparison_reports_loading_series_and_graph_progress(self) -> None:
        def result(engine_name: str) -> EngineResult:
            return EngineResult(
                engine_path=f"/{engine_name}",
                engine_name=engine_name,
                inp_path="model.inp",
                inp_name="model.inp",
                duration_s=1.0,
                peak_memory_mb=1.0,
                exit_code=0,
                rpt_path=None,
                stdout="",
                stderr="",
                error=None,
                out_path=str(example_out_path),
            )

        events = []
        comparisons = compare_all_outputs(
            [result("a"), result("b")],
            progress_callback=events.append,
        )

        self.assertEqual(len(comparisons), 1)
        load_events = [event for event in events if event.phase == "output-load"]
        self.assertEqual(
            [(event.completed, event.total) for event in load_events],
            [(0, 1), (1, 1)],
        )
        series_events = [event for event in events if event.phase == "output-series"]
        graph_events = [event for event in events if event.phase == "output-graph"]
        self.assertTrue(series_events)
        self.assertEqual(series_events[-1].completed, series_events[-1].total)
        self.assertEqual(graph_events, [])
        pair_events = [event for event in events if event.phase == "output-pair"]
        self.assertEqual(
            [event.status for event in pair_events], ["started", "completed"]
        )

    def test_failed_engine_output_is_not_compared(self) -> None:
        failed = EngineResult(
            engine_path="/failed",
            engine_name="failed",
            inp_path="model.inp",
            inp_name="model.inp",
            duration_s=1.0,
            peak_memory_mb=1.0,
            exit_code=1,
            rpt_path=None,
            stdout="",
            stderr="",
            error=None,
            out_path=str(example_out_path),
        )
        succeeded = EngineResult(
            engine_path="/succeeded",
            engine_name="succeeded",
            inp_path="model.inp",
            inp_name="model.inp",
            duration_s=1.0,
            peak_memory_mb=1.0,
            exit_code=0,
            rpt_path=None,
            stdout="",
            stderr="",
            error=None,
            out_path=str(example_out_path),
        )

        self.assertEqual(compare_all_outputs([failed, succeeded]), [])

    def test_unreadable_output_is_skipped_without_aborting_other_results(self) -> None:
        def result(engine_name: str) -> EngineResult:
            return EngineResult(
                engine_path=f"/{engine_name}",
                engine_name=engine_name,
                inp_path="model.inp",
                inp_name="model.inp",
                duration_s=1.0,
                peak_memory_mb=1.0,
                exit_code=0,
                rpt_path=None,
                stdout="",
                stderr="",
                error=None,
                out_path=str(example_out_path),
            )

        with patch(
            "swmm_bench.comparator.extract_output_frame",
            side_effect=ValueError("truncated"),
        ):
            with self.assertWarnsRegex(UserWarning, "Skipping unreadable output"):
                comparisons = compare_all_outputs([result("a"), result("b")])

        self.assertEqual(comparisons, [])

    def test_two_zero_period_outputs_are_not_reported_as_identical(self) -> None:
        with patch(
            "swmm_bench.comparator.extract_output_frame", return_value=DataFrame()
        ):
            with self.assertRaisesRegex(ValueError, "Neither output file"):
                compare_outs("a.out", "b.out", "a", "b", "model.inp", "model.inp")

    def test_zero_period_output_never_queries_series(self) -> None:
        class ZeroPeriodOutput:
            period = 0

            def __init__(self, _path: str, *, preload: bool) -> None:
                self.preload = preload

            def __enter__(self) -> "ZeroPeriodOutput":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

        with patch("swmm_bench.output.Output", ZeroPeriodOutput):
            frame = extract_output_frame("empty.out")

        self.assertTrue(frame.empty)

    def test_empty_categories_are_skipped_and_singletons_are_semantic(self) -> None:
        class SingletonOutput:
            period = 2
            subcatchments: tuple[str, ...] = ()
            nodes = ("N1",)
            links: tuple[str, ...] = ()
            project_size = (0, 1, 0, 0, 0)

            def __init__(self, _path: str, *, preload: bool) -> None:
                self.preload = preload

            def __enter__(self) -> "SingletonOutput":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def node_series(
                self,
                node: str,
                *,
                attribute: None,
                columns: str,
            ) -> DataFrame:
                if node != "N1" or attribute is not None or columns != "attr":
                    raise AssertionError("unexpected node-series query")
                return DataFrame(
                    {"hydraulic_head": [1.0, 2.0]},
                    index=[Timestamp("2020-01-01"), Timestamp("2020-01-01 00:05")],
                )

        with patch("swmm_bench.output.Output", SingletonOutput):
            frame = extract_output_frame(Path("singleton.out"))

        self.assertEqual(
            list(frame.columns),
            [("node", "N1", "hydraulic_head")],
        )
        self.assertEqual(frame.index.name, "datetime")


if __name__ == "__main__":
    unittest.main()
