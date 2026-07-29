from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from swmm.pandas import example_out_path, example_rpt_path
from typer.testing import CliRunner

from swmm_bench.cli import app, test_app as regression_app


class InterfaceWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def _engine(
        self,
        root: Path,
        name: str,
        *,
        fail_on_use: bool = False,
        fail_save_kind: str | None = None,
    ) -> Path:
        engine = root / name
        engine.parent.mkdir(parents=True, exist_ok=True)
        engine.write_text(
            "#!/usr/bin/env python3\n"
            "import re\n"
            "import shutil\n"
            "import sys\n"
            "from pathlib import Path\n"
            "inp, rpt, out = map(Path, sys.argv[1:4])\n"
            "text = inp.read_text(encoding='utf-8')\n"
            f"fail_on_use = {fail_on_use!r}\n"
            f"fail_save_kind = {fail_save_kind!r}\n"
            "if fail_on_use and re.search(r'^USE\\s+', text, re.MULTILINE):\n"
            "    raise SystemExit(7)\n"
            'save = re.search(r\'^SAVE\\s+(\\w+)\\s+"([^"]+)"\', text, re.MULTILINE)\n'
            "if save and save.group(1) == fail_save_kind:\n"
            "    raise SystemExit(9)\n"
            "if save:\n"
            "    (inp.parent / save.group(2)).write_bytes(b'interface-data\\n')\n"
            f"shutil.copyfile({str(example_rpt_path)!r}, rpt)\n"
            f"shutil.copyfile({str(example_out_path)!r}, out)\n",
            encoding="utf-8",
        )
        engine.chmod(0o755)
        return engine

    def test_interface_command_runs_all_comparison_views_and_regenerates_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._engine(root, "source/runswmm")
            target_a = self._engine(root, "epa/runswmm")
            target_b = self._engine(root, "alternate/runswmm")
            output_dir = root / "results"

            result = self.runner.invoke(
                regression_app,
                [
                    "interface",
                    str(source),
                    str(target_a),
                    str(target_b),
                    "--family",
                    "routing",
                    "--name",
                    "interface-test",
                    "--output-dir",
                    str(output_dir),
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            result_dir = output_dir / "interface-test"
            results_json = result_dir / "results.json"
            data = json.loads(results_json.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], "6")
            self.assertEqual(len(data["engine_results"]), 5)
            self.assertEqual(len(data["comparisons"]), 4)
            self.assertEqual(len(data["output_comparisons"]), 4)
            self.assertEqual(
                Counter(item["inp_path"] for item in data["comparisons"]),
                {
                    "interface://routing/consumer": 1,
                    "interface://routing/baseline": 1,
                    "interface://routing/self/0": 1,
                    "interface://routing/self/1": 1,
                },
            )
            consumer_results = [
                item
                for item in data["engine_results"]
                if item["inp_path"] == "interface://routing/consumer"
            ]
            self.assertEqual(
                {item["engine_name"] for item in consumer_results},
                {"epa/runswmm", "alternate/runswmm"},
            )
            self.assertEqual(
                len({item["rpt_path"] for item in consumer_results}),
                2,
            )
            family = data["interface_families"][0]
            self.assertEqual(family["family"], "routing")
            self.assertEqual(family["artifact"]["size_bytes"], 15)
            self.assertEqual(
                family["artifact"]["sha256"],
                hashlib.sha256(b"interface-data\n").hexdigest(),
            )
            self.assertTrue(Path(family["artifact"]["path"]).is_file())

            html = (result_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("swmm-test interface report", html)
            self.assertIn("Routing interface", html)
            self.assertIn("Interface consumer", html)
            self.assertIn("Direct baseline", html)
            self.assertIn("Interface vs direct", html)
            self.assertIn("SHA-256", html)

            regenerated = result_dir / "regenerated.html"
            report_result = self.runner.invoke(
                app,
                ["report", str(results_json), "--output", str(regenerated)],
            )
            self.assertEqual(report_result.exit_code, 0, report_result.output)
            self.assertIn(
                "Routing interface",
                regenerated.read_text(encoding="utf-8"),
            )

    def test_consumer_failure_continues_baseline_and_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._engine(root, "source-swmm")
            target = self._engine(root, "target-swmm", fail_on_use=True)
            output_dir = root / "results"

            result = self.runner.invoke(
                regression_app,
                [
                    "interface",
                    str(source),
                    str(target),
                    "--family",
                    "routing",
                    "--name",
                    "consumer-failure",
                    "--output-dir",
                    str(output_dir),
                    "--no-html",
                ],
            )

            self.assertEqual(result.exit_code, 1, result.output)
            data = json.loads(
                (output_dir / "consumer-failure" / "results.json").read_text(
                    encoding="utf-8"
                )
            )
            by_identity = {item["inp_path"]: item for item in data["engine_results"]}
            self.assertEqual(
                by_identity["interface://routing/consumer"]["exit_code"], 7
            )
            self.assertEqual(
                by_identity["interface://routing/baseline"]["exit_code"], 0
            )

    def test_generator_failure_skips_dependents_and_continues_families(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = self._engine(
                root,
                "source-swmm",
                fail_save_kind="RAINFALL",
            )
            target = self._engine(root, "target-swmm")
            output_dir = root / "results"

            result = self.runner.invoke(
                regression_app,
                [
                    "interface",
                    str(source),
                    str(target),
                    "--family",
                    "rainfall",
                    "--family",
                    "routing",
                    "--name",
                    "generator-failure",
                    "--output-dir",
                    str(output_dir),
                    "--no-html",
                ],
            )

            self.assertEqual(result.exit_code, 1, result.output)
            data = json.loads(
                (output_dir / "generator-failure" / "results.json").read_text(
                    encoding="utf-8"
                )
            )
            families = {item["family"]: item for item in data["interface_families"]}
            self.assertIsNone(families["rainfall"]["artifact"])
            self.assertIsNotNone(families["routing"]["artifact"])
            identities = {item["inp_path"] for item in data["engine_results"]}
            self.assertNotIn("interface://rainfall/consumer", identities)
            self.assertIn("interface://routing/consumer", identities)
            self.assertIn("interface://routing/baseline", identities)


if __name__ == "__main__":
    unittest.main()
