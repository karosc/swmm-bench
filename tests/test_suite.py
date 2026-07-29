from __future__ import annotations

import unittest

from swmm_bench.suite import (
    BENCHMARK_SUITE_NAME,
    SuiteSelectionError,
    catalog_models,
    categories,
    materialize_models,
    select_models,
)


class SuiteTests(unittest.TestCase):
    def test_catalog_has_expected_categories_and_models(self) -> None:
        models = catalog_models()

        self.assertEqual(len(models), 21)
        self.assertEqual(
            categories(),
            (
                "complex",
                "controls",
                "hydraulics",
                "hydrology",
                "routing",
                "use_interfaces",
                "water-quality",
            ),
        )
        self.assertEqual(
            [model.relative_path for model in models],
            sorted(model.relative_path for model in models),
        )

    def test_selects_all_category_or_exact_model(self) -> None:
        self.assertEqual(len(select_models()), 21)
        self.assertEqual(len(select_models(category="hydrology")), 5)

        selected = select_models(model="routing/kinwave-routing_kinwave.inp")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].category, "routing")
        self.assertEqual(
            selected[0].identity,
            "bundled://regression-suite/routing/kinwave-routing_kinwave.inp",
        )

    def test_rejects_invalid_or_conflicting_selection(self) -> None:
        with self.assertRaisesRegex(
            SuiteSelectionError, "either a category or a model"
        ):
            select_models(
                category="hydrology", model="hydrology/lid-example_lid_rb.inp"
            )
        with self.assertRaisesRegex(SuiteSelectionError, "Valid categories"):
            select_models(category="not-a-category")
        with self.assertRaisesRegex(SuiteSelectionError, "Unknown suite model"):
            select_models(model="../hydrology/lid-example_lid_rb.inp")

    def test_materialization_copies_water_quality_sidecar(self) -> None:
        selected = select_models(category="water-quality")

        with materialize_models(selected) as materialized:
            self.assertEqual(len(materialized), 2)
            self.assertTrue(
                (materialized[0].inp_path.parent / "events_example.dat").is_file()
            )
            for item in materialized:
                self.assertTrue(item.inp_path.is_file())
                model_text = item.inp_path.read_text(encoding="utf-8")
                self.assertIn('"events_example.dat"', model_text)
                self.assertNotIn("D:\\SWMMandSoftware", model_text)

    def test_benchmarks_keep_original_complex_simulation_periods(self) -> None:
        benchmark_models = catalog_models(BENCHMARK_SUITE_NAME)
        self.assertEqual(
            [model.relative_path for model in benchmark_models],
            ["complex/gw-events-wq.inp", "complex/rtk.inp"],
        )

        with materialize_models(benchmark_models) as materialized:
            model_text = {
                item.model.relative_path: item.inp_path.read_text(encoding="utf-8")
                for item in materialized
            }
            self.assertIn(
                "START_DATE           01/01/1975",
                model_text["complex/gw-events-wq.inp"],
            )
            self.assertIn(
                "END_DATE             06/02/2015",
                model_text["complex/rtk.inp"],
            )
            self.assertTrue(
                (materialized[0].inp_path.parent / "rainfall-data.dat").is_file()
            )

        regression_models = select_models(category="complex")
        with materialize_models(regression_models) as materialized:
            model_text = {
                item.model.relative_path: item.inp_path.read_text(encoding="utf-8")
                for item in materialized
            }
            self.assertIn(
                "START_DATE           08/07/1989",
                model_text["complex/gw-events-wq.inp"],
            )
            self.assertIn(
                "END_DATE             05/31/2015",
                model_text["complex/rtk.inp"],
            )


if __name__ == "__main__":
    unittest.main()
