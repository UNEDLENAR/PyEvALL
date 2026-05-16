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
Double-implementation tests for legacy classification metric suites.
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


@pytest.mark.classification
class TestClassificationMetrics:
    BASE_DIR = Path(__file__).resolve().parent / "resources" / "metrics" / "double_imp" / "classification"
    V1_DIR = BASE_DIR / "V1_old"
    V2_PYEVALL_DIR = BASE_DIR / "V2_pyevall"
    REPORT_DIR = Path(__file__).resolve().parent / "reports"
    REPORT_CLASSIFICATION_DIR = REPORT_DIR / "classification"
    REPORT_HTML = REPORT_DIR / "test_classification_metrics_report.html"

    V1_EXPECTED_FILE = V1_DIR / "expected_results.txt"
    V1_GOLD_DIR = V1_DIR / "gold"
    V1_PRED_DIR = V1_DIR / "predictions"
    V2_EXPECTED_FILE = V2_PYEVALL_DIR / "expected_results.txt"
    V2_GOLD_DIR = V2_PYEVALL_DIR / "gold"
    V2_PRED_DIR = V2_PYEVALL_DIR / "predictions"
    V2_HIERARCHY_FILE = V2_PYEVALL_DIR / "hierarchy.json"
    V2_HIERARCHY_TEST_CASES = {"HierarchyI", "Hierarchy2"}
    V2_MULTILABEL_TEST_CASES = {"MulticlassGold", "MulticlassSys", "HierarchyI"}
    V2_MULTILABEL_UNSUPPORTED_METRICS = {
        MetricFactory.Accuracy.value,
        MetricFactory.SystemPrecision.value,
        MetricFactory.Kappa.value,
    }
    V2_HIERARCHY_UNSUPPORTED_METRICS = {
        MetricFactory.Accuracy.value,
        MetricFactory.Precision.value,
        MetricFactory.Recall.value,
        MetricFactory.FMeasure.value,
        MetricFactory.SystemPrecision.value,
        MetricFactory.Kappa.value,
    }
    V2_FMEASURE_ALPHA_DISCREPANCY_CASES = {"MinCat", "MulticlassGold", "MulticlassSys"}

    COMPARISON_TOLERANCE = 1e-3

    METRIC_MAP = {
        "ACC": MetricFactory.Accuracy.value,
        "SysPre": MetricFactory.SystemPrecision.value,
        "Kappa": MetricFactory.Kappa.value,
        "ICM": MetricFactory.ICM.value,
        "Precision_TRUE": MetricFactory.Precision.value,
        "Recall_TRUE": MetricFactory.Recall.value,
        "Fmeasure_TRUE": MetricFactory.FMeasure.value,
        "Precision_A": MetricFactory.Precision.value,
        "Recall_A": MetricFactory.Recall.value,
        "Fmeasure_A": MetricFactory.FMeasure.value,
        "Precision_B": MetricFactory.Precision.value,
        "Recall_B": MetricFactory.Recall.value,
        "Fmeasure_B": MetricFactory.FMeasure.value,
    }

    CLASS_METRIC_TARGET = {
        "Precision_TRUE": "TRUE",
        "Recall_TRUE": "TRUE",
        "Fmeasure_TRUE": "TRUE",
        "Precision_A": "A",
        "Recall_A": "A",
        "Fmeasure_A": "A",
        "Precision_B": "B",
        "Recall_B": "B",
        "Fmeasure_B": "B",
    }

    METRICS_TO_EVALUATE = [
        MetricFactory.Accuracy.value,
        MetricFactory.Precision.value,
        MetricFactory.Recall.value,
        MetricFactory.FMeasure.value,
        MetricFactory.SystemPrecision.value,
        MetricFactory.Kappa.value,
        MetricFactory.ICM.value,
    ]

    def _init_run_report_dir(self, suite_name):
        # Create a timestamped directory for raw PyEvALL reports.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.REPORT_CLASSIFICATION_DIR / f"{suite_name}_{timestamp}"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            fallback_base = Path(tempfile.gettempdir()) / "pyevall_reports" / "classification"
            run_dir = fallback_base / f"{suite_name}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
        self._run_report_dir = run_dir

    def _save_raw_pyevall_report(self, testcase, report, report_prefix=""):
        # Persist the raw PyEvALL JSON report for one evaluated test case.
        if not hasattr(self, "_run_report_dir"):
            self._init_run_report_dir("classification")
        testcase_safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in testcase)
        prefix_safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in report_prefix)
        name = f"{testcase_safe}.json" if not prefix_safe else f"{prefix_safe}__{testcase_safe}.json"
        (self._run_report_dir / name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def _parse_value(self, raw_value):
        # Convert legacy textual values into comparable numeric or None values.
        raw_value = raw_value.strip()
        if raw_value in ("", "-", "UNK"):
            return None
        return float(raw_value.replace(",", "."))

    def _parse_v1_expected_rows(self):
        # Parse the V1 legacy classification expected-results table.
        lines = [line.rstrip("\n") for line in self.V1_EXPECTED_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
        header = [token.strip() for token in lines[0].split("\t")]
        rows = []
        for line in lines[1:]:
            tokens = [token.strip() for token in line.split("\t")]
            if len(tokens) < 2 or not Path(tokens[0]).name.startswith("SYS"):
                continue
            testcase = tokens[1]
            for legacy_metric, raw_value in zip(header[3:], tokens[3:]):
                if legacy_metric not in self.METRIC_MAP:
                    continue
                rows.append(
                    {
                        "test_case": testcase,
                        "legacy_metric": legacy_metric,
                        "metric": self.METRIC_MAP[legacy_metric],
                        "class": self.CLASS_METRIC_TARGET.get(legacy_metric, ""),
                        "legacy_value": self._parse_value(raw_value),
                    }
                )
        return rows

    def _parse_v2_expected_rows(self):
        # Parse the V2 classification expected-results table.
        header = None
        rows = []
        for line in self.V2_EXPECTED_FILE.read_text(encoding="utf-8").splitlines():
            raw = line.rstrip("\n")
            if not raw.strip() or raw.startswith("WARNING:"):
                continue
            tokens = [token.strip() for token in raw.split("\t")]
            if "ACC" in tokens and "SysPre" in tokens:
                header = tokens
                continue
            if header is None or tokens[0] != "SysFileTestClassification.csv":
                continue
            testcase = tokens[1]
            for legacy_metric, raw_value in zip(header[3:], tokens[3:]):
                if legacy_metric not in self.METRIC_MAP:
                    continue
                rows.append(
                    {
                        "test_case": testcase,
                        "legacy_metric": legacy_metric,
                        "metric": self.METRIC_MAP[legacy_metric],
                        "class": self.CLASS_METRIC_TARGET.get(legacy_metric, ""),
                        "legacy_value": self._parse_value(raw_value),
                    }
                )
        return rows

    def _extract_metric_value(self, report, metric_name, class_name=""):
        # Read a metric value from a PyEvALL report.
        metric_report = report.get(PyEvALLReport.METRIC_TAG, {}).get(metric_name, {})
        results = metric_report.get(PyEvALLReport.RESULTS_TAG, {})
        if class_name:
            for testcase_result in results.get(PyEvALLReport.TEST_CASES_TAG, []):
                classes = testcase_result.get(PyEvALLReport.CLASSES_TAG, {})
                if class_name in classes:
                    return classes[class_name]
            return None
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

    def _evaluate_case(self, gold_path, pred_path, testcase, tmp_path, report_prefix):
        # Execute PyEvALL for one gold/prediction pair.
        if pred_path.exists():
            pred_eval_path = pred_path
        else:
            pred_eval_path = tmp_path / f"empty_pred_{testcase}.json"
            pred_eval_path.write_text("[]", encoding="utf-8")

        params = {
            PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
            PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE,
        }
        if testcase in self.V2_HIERARCHY_TEST_CASES:
            params[PyEvALLUtils.PARAM_HIERARCHY] = json.loads(self.V2_HIERARCHY_FILE.read_text(encoding="utf-8"))
        report = PyEvALLEvaluation().evaluate(str(pred_eval_path), str(gold_path), self.METRICS_TO_EVALUATE, **params).report
        self._save_raw_pyevall_report(testcase, report, report_prefix=report_prefix)
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

    def _not_applicable_context_reason(self, expected, report_prefix):
        # Explain comparisons skipped because PyEvALL does not define that context.
        if report_prefix != "V2_pyevall":
            return None

        testcase = expected["test_case"]
        metric = expected["metric"]
        if testcase in self.V2_HIERARCHY_TEST_CASES and metric in self.V2_HIERARCHY_UNSUPPORTED_METRICS:
            return "PyEvALL only applies the hierarchy to ICM; this metric is not compared as a hierarchical metric."
        if testcase in self.V2_MULTILABEL_TEST_CASES and metric in self.V2_MULTILABEL_UNSUPPORTED_METRICS:
            return "PyEvALL does not define this metric for multilabel classification; the legacy value is not compared."
        if testcase in self.V2_MULTILABEL_TEST_CASES and metric == MetricFactory.ICM.value and testcase not in self.V2_HIERARCHY_TEST_CASES:
            return "PyEvALL does not currently implement ICM for multilabel classification without a hierarchy; the legacy value is not compared."
        return None

    def _comparison_note(self, row, report_prefix):
        # Add explanatory notes for accepted legacy/PyEvALL discrepancies.
        if (
            report_prefix == "V2_pyevall"
            and row["metric"] == MetricFactory.FMeasure.value
            and row["test_case"] in self.V2_FMEASURE_ALPHA_DISCREPANCY_CASES
            and row["legacy_metric"] == "Fmeasure_B"
        ):
            return "legacy uses alpha=0.2; PyEvALL uses alpha=0.5"
        if (
            report_prefix == "V1_old"
            and row["test_case"] == "25"
            and row["metric"] == MetricFactory.Precision.value
            and row["legacy_metric"] == "Precision_TRUE"
        ):
            return "TRUE is only in predictions; PyEvALL reports class precision only for classes present in gold"
        return None

    def _testcase_sort_key(self, testcase):
        # Sort numeric test-case names before textual names.
        if str(testcase).isdigit():
            return (0, int(testcase))
        return (1, str(testcase).lower())

    def _build_rows(self, expected_rows, gold_dir, pred_dir, tmp_path, report_prefix, v1_numbered=False):
        # Build comparison rows by evaluating each expected test case.
        report_cache = {}
        rows = []
        for expected in expected_rows:
            testcase = expected["test_case"]
            if testcase not in report_cache:
                if v1_numbered:
                    gold_path = gold_dir / f"GOLD{testcase}.txt"
                    pred_path = pred_dir / f"SYS{testcase}.txt"
                else:
                    gold_path = gold_dir / f"{testcase}.json"
                    pred_path = pred_dir / f"{testcase}.json"
                report_cache[testcase] = self._evaluate_case(gold_path, pred_path, testcase, tmp_path, report_prefix)

            report = report_cache[testcase]
            pyevall_value = self._extract_metric_value(report, expected["metric"], expected["class"])
            status, pyevall_error = self._extract_status_and_errors(report, expected["metric"])
            not_applicable_reason = self._not_applicable_context_reason(expected, report_prefix)
            if not_applicable_reason is None:
                comparison, diff = self._compare_values(expected["legacy_value"], pyevall_value)
            else:
                comparison, diff = "NOT_APPLICABLE_CONTEXT", None
                pyevall_error = " | ".join(part for part in (pyevall_error, not_applicable_reason) if part)
            row = {
                **expected,
                "pyevall_value": pyevall_value,
                "abs_diff": diff,
                "status": status,
                "comparison": comparison,
                "pyevall_error": pyevall_error,
            }
            row["comparison_note"] = self._comparison_note(row, report_prefix)
            rows.append(
                row
            )
        return rows

    def _write_html_report(self, sections):
        # Render the detailed comparison rows as an HTML report.
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        all_rows = [row for _, _, rows in sections for row in rows]
        total = len(all_rows)
        matches = sum(1 for row in all_rows if row["comparison"] == "MATCH")
        mismatches = sum(1 for row in all_rows if row["comparison"] == "MISMATCH")
        no_pyevall = sum(1 for row in all_rows if row["comparison"] == "NO_PYEVALL_VALUE")
        unexpected = sum(1 for row in all_rows if row["comparison"] == "UNEXPECTED_PYEVALL_VALUE")
        not_applicable = sum(1 for row in all_rows if row["comparison"] == "NOT_APPLICABLE_CONTEXT")

        html = [
            "<!doctype html>",
            "<html lang='en'>",
            "<head><meta charset='utf-8'><title>Classification Metrics Comparison</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; background: #f8fbff; color: #26313f; margin: 20px; }",
            "h1 { color: #24466f; }",
            ".summary, .legend { background: #eaf3ff; border: 1px solid #bdd7f5; padding: 10px; border-radius: 8px; margin-top: 10px; }",
            "table { border-collapse: collapse; width: 100%; margin-top: 14px; background: #ffffff; }",
            "th { background: #d7e9ff; border: 1px solid #bad2ed; padding: 8px; }",
            "td { border: 1px solid #d7e2ef; padding: 7px; vertical-align: top; }",
            ".row-header { background: #eef6ff; font-weight: 600; }",
            ".match { background: #dff6e6; color: #1d6b35; font-weight: 700; }",
            ".mismatch, .no-pyevall { background: #ffe0e0; color: #a12727; font-weight: 700; }",
            ".unexpected { background: #ffe8d9; color: #8a4b1f; font-weight: 700; }",
            ".not-applicable { background: #eeeeee; color: #555555; font-weight: 700; }",
            ".separator td { border-top: 3px double #8f8f8f; background: #f8fbff; height: 4px; padding: 0; }",
            ".error-cell { max-width: 520px; word-break: break-word; font-size: 12px; color: #6f1d1d; }",
            ".tab-pane { display: none; } .tab-pane.active { display: block; }",
            ".tab-btn { background: #e7f1ff; border: 1px solid #bad2ed; padding: 8px 12px; cursor: pointer; margin: 10px 6px 0 0; border-radius: 6px; }",
            ".tab-btn.active { background: #cfe4ff; font-weight: 700; }",
            "</style>",
            "<script>function openTab(tabId, btn){document.querySelectorAll('.tab-pane').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active'));document.getElementById(tabId).classList.add('active');btn.classList.add('active');}</script>",
            "</head><body>",
            "<h1>Classification detailed comparison report (legacy vs pyevall)</h1>",
            "<div class='summary'>",
            f"<div><strong>Generated:</strong> {datetime.now().isoformat(timespec='seconds')}</div>",
            f"<div><strong>Total:</strong> {total} | <strong>MATCH:</strong> {matches} | <strong>MISMATCH:</strong> {mismatches} | <strong>NO_PYEVALL_VALUE:</strong> {no_pyevall} | <strong>UNEXPECTED_PYEVALL_VALUE:</strong> {unexpected} | <strong>NOT_APPLICABLE_CONTEXT:</strong> {not_applicable}</div>",
            "</div>",
            "<div class='legend'><strong>Legend:</strong> MATCH = same values within tolerance. MISMATCH = different values. NO_PYEVALL_VALUE = PyEvALL did not return value. UNEXPECTED_PYEVALL_VALUE = legacy has UNK/- but PyEvALL returned a value. NOT_APPLICABLE_CONTEXT = legacy defines a value for a context that PyEvALL does not compare for that metric.</div>",
        ]
        for i, (tab_id, tab_label, _) in enumerate(sections):
            active = " active" if i == 0 else ""
            html.append(f"<button class='tab-btn{active}' onclick=\"openTab('{tab_id}', this)\">{tab_label}</button>")
        for i, (tab_id, _, rows) in enumerate(sections):
            active = " active" if i == 0 else ""
            html.append(f"<div id='{tab_id}' class='tab-pane{active}'>")
            html.append("<table><thead><tr><th>metric</th><th>test_case</th><th>description</th><th>legacy_value</th><th>pyevall_value</th><th>abs_diff</th><th>status</th><th>comparison</th><th>pyevall_error</th></tr></thead><tbody>")
            previous_metric = None
            sorted_rows = sorted(rows, key=lambda row: (row["metric"], row["legacy_metric"], row["class"], self._testcase_sort_key(row["test_case"])))
            for row in sorted_rows:
                if previous_metric is not None and row["metric"] != previous_metric:
                    html.append("<tr class='separator'><td colspan='9'></td></tr>")
                previous_metric = row["metric"]
                comparison_class = {
                    "MATCH": "match",
                    "MISMATCH": "mismatch",
                    "NO_PYEVALL_VALUE": "no-pyevall",
                    "UNEXPECTED_PYEVALL_VALUE": "unexpected",
                    "NOT_APPLICABLE_CONTEXT": "not-applicable",
                }.get(row["comparison"], "")
                description = f"legacy={row['legacy_metric']}"
                if row["class"]:
                    description += f"; class={row['class']}"
                if row.get("comparison_note"):
                    description += f" ({row['comparison_note']})"
                html.append(
                    "<tr>"
                    f"<td class='row-header'>{row['metric']}</td>"
                    f"<td class='row-header'>{row['test_case']}</td>"
                    f"<td>{description}</td>"
                    f"<td>{row['legacy_value']}</td><td>{row['pyevall_value']}</td>"
                    f"<td>{row['abs_diff']}</td><td>{row['status']}</td><td class='{comparison_class}'>{row['comparison']}</td>"
                    f"<td class='error-cell'>{row['pyevall_error']}</td></tr>"
                )
            html.extend(["</tbody></table></div>"])
        html.extend(["</body></html>"])
        self.REPORT_HTML.write_text("\n".join(html) + "\n", encoding="utf-8")

    def test_classification_legacy_compare_with_pyevall_and_generate_html(self, tmp_path):
        # Compare classification legacy results with PyEvALL and generate the HTML report.
        self._init_run_report_dir("classification_legacy")
        rows_v1 = self._build_rows(self._parse_v1_expected_rows(), self.V1_GOLD_DIR, self.V1_PRED_DIR, tmp_path, "V1_old", v1_numbered=True)
        rows_v2 = self._build_rows(self._parse_v2_expected_rows(), self.V2_GOLD_DIR, self.V2_PRED_DIR, tmp_path, "V2_pyevall")
        self._write_html_report(
            [
                ("tab_v1", "Classification V1 old", rows_v1),
                ("tab_v2", "Classification V2 PyEvALL format", rows_v2),
            ]
        )
        assert self.REPORT_HTML.exists()
        report_text = self.REPORT_HTML.read_text(encoding="utf-8")
        assert "Classification V1 old" in report_text
        assert "Classification V2 PyEvALL format" in report_text
