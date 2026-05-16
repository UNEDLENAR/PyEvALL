# ============================================================================== 
#
# Copyright (c) 2026 - Permission is granted for use and modification of this file for research, non-commercial purposes. 
#
# This work has been financed by the European Union (NextGenerationEU funds) through the
# "Plan de Recuperacion, Transformacion y Resiliencia", by the Ministry of Economic Affairs and Digital Transformation and by the UNED University.
#
# The work has been developed within the project "ESPACIO DE OBSERVACION DE INTELIGENCIA ARTIFICIAL (IA) EN ESPANOL", in the framework of the
# Convenio C039/21-OT between the public entity RED.ES, M.P. and the UNIVERSIDAD NACIONAL DE EDUCACION A DISTANCIA (UNED).
#
# @author Jorge Carrillo-de-Albornoz <jcalbornoz@lsi.uned.es> 
# 
# Licensed under the European Union Public License (EUPL) v.1.1. (the "License"); 
# you may not use this file except in compliance with the License. 
# 
# You may obtain a copy of the License at 
#
#         https://commission.europa.eu/content/european-union-public-licence_en 
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed 
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for 
# the specific language governing permissions and limitations under the License. 
#
# ============================================================================== 

"""
Double-implementation tests for legacy ranking metric suite.
"""

import json
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from pyevall.reports.reports import PyEvALLReport
from pyevall.utils.utils import PyEvALLUtils


@pytest.mark.ranking
class TestRankingMetrics:
    BASE_DIR = Path(__file__).resolve().parent / "resources" / "metrics" / "double_imp" / "ranking" / "V1_old"
    GOLD_DIR = BASE_DIR / "gold"
    PRED_DIR = BASE_DIR / "predictions"
    EXPECTED_DIR = BASE_DIR / "expected"
    REPORT_DIR = Path(__file__).resolve().parent / "reports"
    REPORT_RANKING_DIR = REPORT_DIR / "ranking"
    REPORT_HTML = REPORT_DIR / "test_ranking_metrics_report.html"

    COMPARISON_TOLERANCE = 1e-3

    LEGACY_METRIC_MAP = {
        "precisionatk": MetricFactory.PrecisionAtK.value,
        "rprecision": MetricFactory.RPrecision.value,
        "mainreciprocalrank": MetricFactory.MRR.value,
        "map": MetricFactory.MAP.value,
        "dcg": MetricFactory.DCG.value,
        "ndcg": MetricFactory.nDCG.value,
        "err": MetricFactory.ERR.value,
        "rbp": MetricFactory.RBP.value,
    }

    METRICS_TO_EVALUATE = [
        MetricFactory.PrecisionAtK.value,
        MetricFactory.RPrecision.value,
        MetricFactory.MRR.value,
        MetricFactory.MAP.value,
        MetricFactory.DCG.value,
        MetricFactory.nDCG.value,
        MetricFactory.ERR.value,
        MetricFactory.RBP.value,
    ]

    def _init_run_report_dir(self, suite_name):
        # Create a timestamped directory for raw PyEvALL reports.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.REPORT_RANKING_DIR / f"{suite_name}_{timestamp}"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            fallback_base = Path(tempfile.gettempdir()) / "pyevall_reports" / "ranking"
            run_dir = fallback_base / f"{suite_name}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
        self._run_report_dir = run_dir

    def _save_raw_pyevall_report(self, testcase, report):
        # Persist the raw PyEvALL JSON report for one evaluated test case.
        if not hasattr(self, "_run_report_dir"):
            self._init_run_report_dir("ranking")
        testcase_safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in testcase)
        (self._run_report_dir / f"{testcase_safe}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def _parse_expected_file(self, path):
        # Implement helper logic for _parse_expected_file.
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            legacy_metric, raw_value = [token.strip() for token in raw.split("\t", 1)]
            if legacy_metric not in self.LEGACY_METRIC_MAP:
                continue
            rows.append(
                {
                    "legacy_metric": legacy_metric,
                    "metric": self.LEGACY_METRIC_MAP[legacy_metric],
                    "legacy_value": None if raw_value in ("", "-", "UNK") else float(raw_value.replace(",", ".")),
                }
            )
        return rows

    def _extract_metric_value(self, report, metric_name):
        # Read a metric value from a PyEvALL report.
        metric_report = report.get(PyEvALLReport.METRIC_TAG, {}).get(metric_name, {})
        results = metric_report.get(PyEvALLReport.RESULTS_TAG, {})
        return results.get(PyEvALLReport.AVERAGE_PER_TC_TAG, results.get(PyEvALLReport.AVERAGE_TAG))

    def _extract_status_and_errors(self, report, metric_name):
        # Read metric status and diagnostic messages from a PyEvALL report.
        metric_report = report.get(PyEvALLReport.METRIC_TAG, {}).get(metric_name, {})
        status = metric_report.get(PyEvALLReport.STATUS_TAG, "UNKNOWN")
        errors = []
        for err in metric_report.get("errors", []):
            if isinstance(err, dict):
                errors.append(f"{err.get('name', '')}: {err.get('description', '')}".strip(": "))
            else:
                errors.append(str(err))
        preconditions = metric_report.get(PyEvALLReport.PRECONDITIONS_TAG, {})
        if isinstance(preconditions, dict):
            for pre_key, pre_val in preconditions.items():
                if isinstance(pre_val, dict):
                    errors.append(f"{pre_key} ({pre_val.get('status', '')}): {pre_val.get('description', '')}".strip(": "))
                else:
                    errors.append(str(pre_key))
        return status, " | ".join(e for e in errors if e)

    def _evaluate_case(self, testcase):
        # Execute PyEvALL for one gold/prediction pair.
        params = {
            PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
            PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE,
            "p_param": 0.5,
        }
        pred_path = self.PRED_DIR / f"{testcase}.txt"
        gold_path = self.GOLD_DIR / f"GOLD{testcase[3:]}.txt"
        report = PyEvALLEvaluation().evaluate(str(pred_path), str(gold_path), self.METRICS_TO_EVALUATE, **params).report
        self._save_raw_pyevall_report(testcase, report)
        return report

    def _compare_values(self, expected_value, pyevall_value):
        # Compare legacy and PyEvALL values using the configured tolerance.
        if expected_value is None and pyevall_value is None:
            return "MATCH", None
        if expected_value is None and pyevall_value is not None:
            return "UNEXPECTED_PYEVALL_VALUE", None
        if pyevall_value is None:
            return "NO_PYEVALL_VALUE", None
        diff = abs(float(pyevall_value) - float(expected_value))
        if math.isnan(diff):
            return "MATCH", diff
        return ("MATCH" if diff <= self.COMPARISON_TOLERANCE else "MISMATCH"), diff

    def _build_rows(self):
        # Build comparison rows by evaluating each expected test case.
        rows = []
        for expected_file in sorted(self.EXPECTED_DIR.glob("SYS*.txt"), key=lambda p: int(p.stem[3:])):
            testcase = expected_file.stem
            report = self._evaluate_case(testcase)
            for expected in self._parse_expected_file(expected_file):
                pyevall_value = self._extract_metric_value(report, expected["metric"])
                status, pyevall_error = self._extract_status_and_errors(report, expected["metric"])
                comparison, diff = self._compare_values(expected["legacy_value"], pyevall_value)
                rows.append(
                    {
                        "test_case": testcase,
                        **expected,
                        "pyevall_value": pyevall_value,
                        "abs_diff": diff,
                        "status": status,
                        "comparison": comparison,
                        "pyevall_error": pyevall_error,
                    }
                )
        return rows

    def _write_html_report(self, rows):
        # Render the detailed comparison rows as an HTML report.
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        total = len(rows)
        matches = sum(1 for row in rows if row["comparison"] == "MATCH")
        mismatches = sum(1 for row in rows if row["comparison"] == "MISMATCH")
        no_pyevall = sum(1 for row in rows if row["comparison"] == "NO_PYEVALL_VALUE")
        unexpected = sum(1 for row in rows if row["comparison"] == "UNEXPECTED_PYEVALL_VALUE")

        html = [
            "<!doctype html>",
            "<html lang='en'>",
            "<head><meta charset='utf-8'><title>Ranking Metrics Comparison</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; background: #fbfff9; color: #263523; margin: 20px; }",
            "h1 { color: #315d2d; }",
            ".summary, .legend { background: #eef9e9; border: 1px solid #cae4c2; padding: 10px; border-radius: 8px; margin-top: 10px; }",
            "table { border-collapse: collapse; width: 100%; margin-top: 14px; background: #ffffff; }",
            "th { background: #daf0d2; border: 1px solid #c2dab9; padding: 8px; }",
            "td { border: 1px solid #d7e6d1; padding: 7px; vertical-align: top; }",
            ".row-header { background: #f1faee; font-weight: 600; }",
            ".match { background: #dff6e6; color: #1d6b35; font-weight: 700; }",
            ".mismatch, .no-pyevall { background: #ffe0e0; color: #a12727; font-weight: 700; }",
            ".unexpected { background: #ffe8d9; color: #8a4b1f; font-weight: 700; }",
            ".separator td { border-top: 3px double #8f8f8f; background: #fbfff9; height: 4px; padding: 0; }",
            ".error-cell { max-width: 520px; word-break: break-word; font-size: 12px; color: #6f1d1d; }",
            "</style></head><body>",
            "<h1>Ranking detailed comparison report (legacy vs pyevall)</h1>",
            "<div class='summary'>",
            f"<div><strong>Generated:</strong> {datetime.now().isoformat(timespec='seconds')}</div>",
            f"<div><strong>Total:</strong> {total} | <strong>MATCH:</strong> {matches} | <strong>MISMATCH:</strong> {mismatches} | <strong>NO_PYEVALL_VALUE:</strong> {no_pyevall} | <strong>UNEXPECTED_PYEVALL_VALUE:</strong> {unexpected}</div>",
            "</div>",
            "<div class='legend'><strong>Legend:</strong> MATCH = same values within tolerance. MISMATCH = different values. NO_PYEVALL_VALUE = PyEvALL did not return value. UNEXPECTED_PYEVALL_VALUE = legacy has UNK/- but PyEvALL returned a value.</div>",
            "<table><thead><tr><th>metric</th><th>test_case</th><th>description</th><th>legacy_value</th><th>pyevall_value</th><th>abs_diff</th><th>status</th><th>comparison</th><th>pyevall_error</th></tr></thead><tbody>",
        ]
        previous_case = None
        sorted_rows = sorted(rows, key=lambda row: (row["metric"], int(row["test_case"][3:])))
        previous_metric = None
        for row in sorted_rows:
            if previous_metric is not None and row["metric"] != previous_metric:
                html.append("<tr class='separator'><td colspan='9'></td></tr>")
            previous_metric = row["metric"]
            previous_case = row["test_case"]
            comparison_class = {"MATCH": "match", "MISMATCH": "mismatch", "NO_PYEVALL_VALUE": "no-pyevall", "UNEXPECTED_PYEVALL_VALUE": "unexpected"}.get(row["comparison"], "")
            html.append(
                "<tr>"
                f"<td class='row-header'>{row['metric']}</td>"
                f"<td class='row-header'>{row['test_case']}</td>"
                f"<td>legacy={row['legacy_metric']}</td>"
                f"<td>{row['legacy_value']}</td><td>{row['pyevall_value']}</td><td>{row['abs_diff']}</td>"
                f"<td>{row['status']}</td><td class='{comparison_class}'>{row['comparison']}</td><td class='error-cell'>{row['pyevall_error']}</td>"
                "</tr>"
            )
        html.extend(["</tbody></table></body></html>"])
        self.REPORT_HTML.write_text("\n".join(html) + "\n", encoding="utf-8")

    def test_ranking_legacy_compare_with_pyevall_and_generate_html(self):
        # Compare ranking legacy results with PyEvALL and generate the HTML report.
        self._init_run_report_dir("ranking_legacy")
        rows = self._build_rows()
        self._write_html_report(rows)
        assert self.REPORT_HTML.exists()
        assert "Ranking detailed comparison report" in self.REPORT_HTML.read_text(encoding="utf-8")
