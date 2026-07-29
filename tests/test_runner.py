from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from swmm_bench.runner import _copy_model_tree, _set_threads, run_benchmark


class RunBenchmarkTests(unittest.TestCase):
    def test_nested_output_directory_is_not_copied_into_model_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_directory = root / "model"
            model_directory.mkdir()
            inp_path = model_directory / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            auxiliary_path = model_directory / "rainfall.dat"
            auxiliary_path.write_text("rainfall\n", encoding="utf-8")

            engine_path = root / "fake-swmm"
            engine_path.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text('report\\n', encoding='utf-8')\n"
                "pathlib.Path(sys.argv[3]).write_bytes(b'output')\n",
                encoding="utf-8",
            )
            engine_path.chmod(0o755)

            output_root = model_directory / "benches"
            results = run_benchmark(
                engines=[str(engine_path)],
                inp_files=[inp_path],
                work_dir=output_root,
                benchmark_name="bench",
                timeout=5.0,
            )

            staged_model = (
                output_root / "bench" / engine_path.name / inp_path.name / "model"
            )
            self.assertIsNone(results[0].error)
            self.assertIsNotNone(results[0].out_path)
            self.assertEqual(Path(results[0].out_path).read_bytes(), b"output")
            self.assertEqual(
                Path(results[0].out_path).name,
                "result.out",
            )
            self.assertEqual(
                (staged_model / inp_path.name).read_text(encoding="utf-8"), "[TITLE]\n"
            )
            self.assertEqual(
                (staged_model / auxiliary_path.name).read_text(encoding="utf-8"),
                "rainfall\n",
            )
            self.assertFalse((staged_model / output_root.name).exists())

    def test_duration_comes_from_report_analysis_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inp_path = root / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            engine_path = root / "fake-swmm"
            engine_path.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text(\n"
                "    '  Total elapsed time: 00:00:02\\n', encoding='utf-8'\n"
                ")\n",
                encoding="utf-8",
            )
            engine_path.chmod(0o755)

            result = run_benchmark(
                engines=[str(engine_path)],
                inp_files=[inp_path],
                work_dir=root / "results",
                benchmark_name="bench",
                timeout=5.0,
            )[0]

            self.assertEqual(result.duration_s, 2.0)

    def test_repeat_run_does_not_reuse_stale_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inp_path = root / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            engine_path = root / "fake-swmm"
            engine_path.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text('report\\n', encoding='utf-8')\n"
                "pathlib.Path(sys.argv[3]).write_bytes(b'output')\n",
                encoding="utf-8",
            )
            engine_path.chmod(0o755)
            output_root = root / "results"

            first = run_benchmark(
                engines=[str(engine_path)],
                inp_files=[inp_path],
                work_dir=output_root,
                benchmark_name="bench",
                timeout=5.0,
            )[0]
            self.assertIsNotNone(first.rpt_path)
            self.assertIsNotNone(first.out_path)

            engine_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            engine_path.chmod(0o755)
            second = run_benchmark(
                engines=[str(engine_path)],
                inp_files=[inp_path],
                work_dir=output_root,
                benchmark_name="bench",
                timeout=5.0,
            )[0]

            self.assertIsNone(second.rpt_path)
            self.assertIsNone(second.out_path)
            self.assertEqual(
                second.error, "Engine did not produce a non-empty report file"
            )

    def test_benchmark_preserves_supplied_input_name_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inp_path = root / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            engine_path = root / "fake-swmm"
            engine_path.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                "pathlib.Path(sys.argv[2]).write_text('report\\n', encoding='utf-8')\n",
                encoding="utf-8",
            )
            engine_path.chmod(0o755)

            results = run_benchmark(
                engines=[str(engine_path)],
                inp_files=[inp_path],
                work_dir=root / "results",
                benchmark_name="bench",
                timeout=5.0,
                inp_names={inp_path: "suite/Model.inp"},
                inp_identities={inp_path: "bundled://regression-suite/suite/Model.inp"},
            )

            self.assertEqual(results[0].inp_name, "suite/Model.inp")
            self.assertEqual(
                results[0].inp_path, "bundled://regression-suite/suite/Model.inp"
            )
            self.assertIsNone(results[0].out_path)

    def test_model_copy_does_not_follow_directory_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model_directory = root / "model"
            model_directory.mkdir()
            inp_path = model_directory / "Model.inp"
            inp_path.write_text("[TITLE]\n", encoding="utf-8")
            loop_path = model_directory / "loop"
            try:
                loop_path.symlink_to(model_directory, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"Directory symlinks are unavailable: {exc}")

            staged_model = root / "staged"
            _copy_model_tree(inp_path, staged_model)

            self.assertEqual(
                (staged_model / inp_path.name).read_text(encoding="utf-8"), "[TITLE]\n"
            )
            self.assertTrue((staged_model / loop_path.name).is_dir())
            self.assertEqual(list((staged_model / loop_path.name).iterdir()), [])

    def test_thread_override_does_not_parse_or_rewrite_the_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            inp_path = Path(temporary_directory) / "Model.inp"
            inp_path.write_text(
                "[OPTIONS]\nTHREADS              4\n\n[EVENT]\n01/01/2020 1\n",
                encoding="utf-8",
            )

            _set_threads(inp_path, 2)

            self.assertEqual(
                inp_path.read_text(encoding="utf-8"),
                "[OPTIONS]\nTHREADS              2\n\n[EVENT]\n01/01/2020 1\n",
            )


if __name__ == "__main__":
    unittest.main()
