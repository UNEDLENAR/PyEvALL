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
Double-implementation tests for clustering metrics with two suites:
1. V1_enrique
2. V2_enrique
"""

import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.reports.reports import PyEvALLReport
from pyevall.utils.utils import PyEvALLUtils


@pytest.mark.clustering
class TestClusteringMetrics:
    BASE_DIR = Path(__file__).resolve().parent / "resources" / "metrics" / "double_imp" / "clustering"
    V1_DIR = BASE_DIR / "V1_enrique"
    V2_DIR = BASE_DIR / "V2_enrique"
    REPORT_DIR = Path(__file__).resolve().parent / "reports"
    REPORT_CLUSTERING_DIR = REPORT_DIR / "clustering"
    REPORT_HTML = REPORT_DIR / "test_clustering_metrics_report.html"

    V1_GOLD_SOURCE_FILE = V1_DIR / "ClusGold.csv"
    V1_SYSA_SOURCE_FILE = V1_DIR / "ClusSysA.csv"
    V1_SYSB_SOURCE_FILE = V1_DIR / "ClusSysB.csv"
    V1_RESULTS_FILE = V1_DIR / "results.txt"
    V1_GOLD_PYEVALL_JSON = V1_DIR / "Pyevall_ClusGold.json"
    V1_SYSA_PYEVALL_JSON = V1_DIR / "Pyevall_ClusSysA.json"
    V1_SYSB_PYEVALL_JSON = V1_DIR / "Pyevall_ClusSysB.json"
    V2_GOLD_SOURCE_FILE = V2_DIR / "GoldEstressTestClustering.csv"
    V2_SYS_SOURCE_FILE = V2_DIR / "SysEstressTestClustering.csv"
    V2_RESULTS_FILE = V2_DIR / "Results.txt"
    V2_GOLD_PYEVALL_JSON = V2_DIR / "Pyevall_GoldEstressTestClustering.json"
    V2_SYS_PYEVALL_JSON = V2_DIR / "Pyevall_SysEstressTestClustering.json"

    COMPARISON_TOLERANCE = 1e-3

    METRIC_PURITY = "Purity"
    METRIC_INVERSE_PURITY = "InversePurity"
    METRIC_FMEASURE_P_INV = "FMeasurePurityInversePurity"
    METRIC_FMEASURE_CLUSTERING = "FMeasureClustering"
    METRIC_RAND = "RandStatistics"
    METRIC_JACCARD = "Jaccard"
    METRIC_FOLKES_MALLOWS = "FolkesAndMallows"
    METRIC_RELIABILITY = "ReliabilityBCubed"
    METRIC_SENSITIVITY = "SensitivityBCubed"
    METRIC_NMI_SG = "NMI-SG"
    METRIC_ENTROPY = "Entropy"
    METRIC_CLASS_ENTROPY = "ClassEntropy"
    METRIC_VI = "VariationOfInformation"

    V1_RESULTS_METRIC_MAP = {
        "Purity": METRIC_PURITY,
        "InvPurity": METRIC_INVERSE_PURITY,
        "PurInPurFmeasure": METRIC_FMEASURE_P_INV,
        "ClusFmeasure_0.5": METRIC_FMEASURE_CLUSTERING,
        "CountingPairs_Rand": METRIC_RAND,
        "CountingPairs_Jaccard": METRIC_JACCARD,
        "CountingPairs_FolkesMallows": METRIC_FOLKES_MALLOWS,
        "ClasEntropy": METRIC_CLASS_ENTROPY,
        "ClusEntropy": METRIC_ENTROPY,
        "NMI-SG": METRIC_NMI_SG,
        "ExtCubedPrecision": METRIC_RELIABILITY,
        "ExtCubedRecall": METRIC_SENSITIVITY,
    }

    V2_RESULTS_METRIC_MAP = V1_RESULTS_METRIC_MAP

    def _load_json(self, path):
        # Load a JSON fixture file.
        return json.loads(path.read_text(encoding="utf-8"))

    def _init_run_report_dir(self, suite_name):
        # Create a timestamped directory for raw PyEvALL reports.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.REPORT_CLUSTERING_DIR / f"{suite_name}_{timestamp}"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            fallback_base = Path(tempfile.gettempdir()) / "pyevall_reports" / "clustering"
            run_dir = fallback_base / f"{suite_name}_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)
        self._run_report_dir = run_dir

    def _save_raw_pyevall_report(self, testcase, metric_name, report, report_prefix=""):
        # Persist the raw PyEvALL JSON report for one evaluated test case.
        if not hasattr(self, "_run_report_dir"):
            self._init_run_report_dir("clustering")
        metric_safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in metric_name)
        testcase_safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in testcase)
        prefix_safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in report_prefix)
        name = f"{testcase_safe}__{metric_safe}.json" if not prefix_safe else f"{prefix_safe}__{testcase_safe}__{metric_safe}.json"
        report_file = self._run_report_dir / name
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def _parse_legacy_clustering_tsv_to_pyevall_json(self, tsv_path):
        # Convert legacy clustering TSV-like data into PyEvALL JSON records.
        grouped = {}
        for line in tsv_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            cols = raw.split("\t")
            if len(cols) < 3:
                continue
            testcase = cols[0].strip()
            item_id = cols[1].strip()
            value = cols[2].strip()
            key = (testcase, item_id)
            grouped.setdefault(key, [])
            if value not in grouped[key]:
                grouped[key].append(value)

        records = []
        for testcase, item_id in sorted(grouped.keys(), key=lambda x: (x[0], x[1])):
            records.append(
                {
                    "test_case": testcase,
                    "id": item_id,
                    "value": grouped[(testcase, item_id)],
                }
            )
        return records

    def _ensure_pyevall_json_from_tsv_like(self, source_path, target_path):
        # Create a converted PyEvALL JSON file when it is missing.
        records = self._parse_legacy_clustering_tsv_to_pyevall_json(source_path)
        target_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return records

    def _subset_by_testcase(self, data, testcase):
        # Select fixture records belonging to one test case.
        return [inst for inst in data if inst["test_case"] == testcase]

    def _extract_numeric_value(self, obj):
        # Extract a numeric metric value from a PyEvALL report.
        if isinstance(obj, (int, float)):
            return float(obj)
        if isinstance(obj, dict):
            for key in ("average_per_test_case", "average", "value", "result"):
                value = obj.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            for value in obj.values():
                extracted = self._extract_numeric_value(value)
                if extracted is not None:
                    return extracted
        if isinstance(obj, list):
            for item in obj:
                extracted = self._extract_numeric_value(item)
                if extracted is not None:
                    return extracted
        return None

    def _evaluate_metric_for_testcase(self, tmp_path, testcase, metric_name, metric_params=None, gold_data=None, pred_data=None, report_prefix=""):
        # Evaluate one clustering metric for one test case.
        if gold_data is None or pred_data is None:
            raise ValueError("gold_data and pred_data are required")
        gold_subset = self._subset_by_testcase(gold_data, testcase)
        pred_subset = self._subset_by_testcase(pred_data, testcase)

        gold_tmp = tmp_path / f"gold_{testcase}.json"
        pred_tmp = tmp_path / f"pred_{testcase}.json"
        gold_tmp.write_text(json.dumps(gold_subset, ensure_ascii=False, indent=2), encoding="utf-8")
        pred_tmp.write_text(json.dumps(pred_subset, ensure_ascii=False, indent=2), encoding="utf-8")

        params = {
            PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
            PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE,
        }
        if metric_params:
            params["metric_params"] = {metric_name: metric_params}

        report = PyEvALLEvaluation().evaluate(str(pred_tmp), str(gold_tmp), [metric_name], **params).report
        self._save_raw_pyevall_report(testcase, metric_name, report, report_prefix=report_prefix)
        metric_tag = getattr(PyEvALLReport, "METRIC_TAG", "metrics")
        status_tag = getattr(PyEvALLReport, "STATUS_TAG", "status")
        results_tag = getattr(PyEvALLReport, "RESULTS_TAG", "results")
        avg_tag = getattr(PyEvALLReport, "AVERAGE_PER_TC_TAG", "average_per_test_case")
        metric_report = report.get(metric_tag, {}).get(metric_name, {})
        status = metric_report.get(status_tag, "UNKNOWN")
        value = None
        if results_tag in metric_report and isinstance(metric_report[results_tag], dict):
            results_obj = metric_report[results_tag]
            value = results_obj.get(avg_tag, results_obj.get("average_per_test_case", results_obj.get("average")))
        if value is None:
            value = self._extract_numeric_value(metric_report)
        errors = []
        for err in metric_report.get("errors", []):
            if isinstance(err, dict):
                name = err.get("name", "")
                desc = err.get("description", "")
                errors.append(f"{name}: {desc}".strip(": "))
            else:
                errors.append(str(err))
        preconditions = metric_report.get("preconditions", {})
        if isinstance(preconditions, dict):
            for pre_key, pre_val in preconditions.items():
                if isinstance(pre_val, dict):
                    p_status = pre_val.get("status", "")
                    p_desc = pre_val.get("description", "")
                    errors.append(f"{pre_key} ({p_status}): {p_desc}".strip(": "))
                else:
                    errors.append(str(pre_key))
        error_info = " | ".join(e for e in errors if e).strip()
        return status, value, error_info

    def _parse_v1_expected_results_by_system(self):
        # Parse V1 clustering expected results grouped by system file.
        expected = {"ClusSysA.csv": {}, "ClusSysB.csv": {}}
        header_metrics = None
        started_table = False
        for line in self.V1_RESULTS_FILE.read_text(encoding="utf-8").splitlines():
            raw = line.strip("\n")
            if not raw.strip():
                continue
            if "Purity" in raw and "InvPurity" in raw and "ExtCubedRecall" in raw:
                tokens = [t.strip() for t in raw.split("\t") if t.strip()]
                header_metrics = tokens
                started_table = True
                continue
            if not started_table:
                continue
            if not raw.startswith("./ClusSys"):
                continue

            tokens = [t.strip() for t in raw.split("\t")]
            system_file = Path(tokens[0].strip()).name
            testcase = tokens[1].strip()
            metric_values = [t.strip() for t in tokens[2:] if t.strip()]
            if header_metrics is None or len(metric_values) < len(header_metrics):
                continue

            for idx, legacy_metric_name in enumerate(header_metrics):
                if legacy_metric_name not in self.V1_RESULTS_METRIC_MAP:
                    continue
                metric_name = self.V1_RESULTS_METRIC_MAP[legacy_metric_name]
                raw_value = metric_values[idx]
                if raw_value in ("-", "UNK", ""):
                    parsed_value = None
                else:
                    parsed_value = float(raw_value.replace(",", "."))
                expected.setdefault(system_file, {})
                expected[system_file].setdefault(metric_name, {})
                expected[system_file][metric_name][testcase] = parsed_value

        return expected

    def _parse_v2_expected_results_by_system(self):
        # Parse V2 clustering expected results grouped by system file.
        expected = {"SysEstressTestClustering.csv": {}}
        header_metrics = None
        started_table = False
        for line in self.V2_RESULTS_FILE.read_text(encoding="utf-8").splitlines():
            raw = line.strip("\n")
            if not raw.strip():
                continue
            if "Purity" in raw and "InvPurity" in raw and "ExtCubedRecall" in raw:
                tokens = [t.strip() for t in raw.split("\t") if t.strip()]
                header_metrics = tokens
                started_table = True
                continue
            if not started_table:
                continue
            if not raw.startswith("./PRUEBAS/PRUEBAS_EVALL/CLUSTERING/"):
                continue

            tokens = [t.strip() for t in raw.split("\t")]
            system_file = Path(tokens[0].strip()).name
            testcase = tokens[1].strip()
            metric_values = [t.strip() for t in tokens[2:] if t.strip()]
            if header_metrics is None or len(metric_values) < len(header_metrics):
                continue

            for idx, legacy_metric_name in enumerate(header_metrics):
                if legacy_metric_name not in self.V2_RESULTS_METRIC_MAP:
                    continue
                metric_name = self.V2_RESULTS_METRIC_MAP[legacy_metric_name]
                raw_value = metric_values[idx]
                if raw_value in ("-", "UNK", ""):
                    parsed_value = None
                else:
                    parsed_value = float(raw_value.replace(",", "."))
                expected.setdefault(system_file, {})
                expected[system_file].setdefault(metric_name, {})
                expected[system_file][metric_name][testcase] = parsed_value
        return expected

    def _build_v1_comparison_rows(self, tmp_path, system_file_name, pred_json_path):
        # Build comparison rows for V1 clustering fixtures.
        expected_by_system = self._parse_v1_expected_results_by_system()
        expected_by_metric = expected_by_system.get(system_file_name, {})
        gold_data = self._load_json(self.V1_GOLD_PYEVALL_JSON)
        pred_data = self._load_json(pred_json_path)
        rows = []

        for metric_name, expected_map in sorted(expected_by_metric.items()):
            for testcase, expected_value in sorted(expected_map.items()):
                status, pyevall_value, pyevall_error = self._evaluate_metric_for_testcase(
                    tmp_path, testcase, metric_name, gold_data=gold_data, pred_data=pred_data, report_prefix=f"V1_{system_file_name}"
                )
                if expected_value is None and pyevall_value is None:
                    comparison = "MATCH"
                    diff = None
                elif expected_value is None and pyevall_value is not None:
                    comparison = "UNEXPECTED_PYEVALL_VALUE"
                    diff = None
                elif pyevall_value is None:
                    comparison = "NO_PYEVALL_VALUE"
                    diff = None
                else:
                    diff = abs(pyevall_value - expected_value)
                    comparison = "MATCH" if diff <= self.COMPARISON_TOLERANCE else "MISMATCH"

                rows.append(
                    {
                        "metric": metric_name,
                        "test_case": testcase,
                        "test_case_description": "",
                        "legacy_value": expected_value,
                        "pyevall_value": pyevall_value,
                        "abs_diff": diff,
                        "status": status,
                        "comparison": comparison,
                        "pyevall_error": pyevall_error,
                    }
                )
        return rows

    def _build_v2_comparison_rows(self, tmp_path, system_file_name, pred_json_path):
        # Build comparison rows for V2 clustering fixtures.
        expected_by_system = self._parse_v2_expected_results_by_system()
        expected_by_metric = expected_by_system.get(system_file_name, {})
        gold_data = self._load_json(self.V2_GOLD_PYEVALL_JSON)
        pred_data = self._load_json(pred_json_path)
        rows = []
        for metric_name, expected_map in sorted(expected_by_metric.items()):
            for testcase, expected_value in sorted(expected_map.items()):
                status, pyevall_value, pyevall_error = self._evaluate_metric_for_testcase(
                    tmp_path, testcase, metric_name, gold_data=gold_data, pred_data=pred_data, report_prefix=f"V2_{system_file_name}"
                )
                if expected_value is None and pyevall_value is None:
                    comparison = "MATCH"
                    diff = None
                elif expected_value is None and pyevall_value is not None:
                    comparison = "UNEXPECTED_PYEVALL_VALUE"
                    diff = None
                elif pyevall_value is None:
                    comparison = "NO_PYEVALL_VALUE"
                    diff = None
                else:
                    diff = abs(pyevall_value - expected_value)
                    comparison = "MATCH" if diff <= self.COMPARISON_TOLERANCE else "MISMATCH"
                rows.append(
                    {
                        "metric": metric_name,
                        "test_case": testcase,
                        "test_case_description": "",
                        "legacy_value": expected_value,
                        "pyevall_value": pyevall_value,
                        "abs_diff": diff,
                        "status": status,
                        "comparison": comparison,
                        "pyevall_error": pyevall_error,
                    }
                )
        return rows

    def _write_html_report(self, sections):
        # Render the detailed comparison rows as an HTML report.
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        all_rows = []
        for _, _, rows in sections:
            all_rows.extend(rows)
        total = len(all_rows)
        matches = sum(1 for r in all_rows if r["comparison"] == "MATCH")
        mismatches = sum(1 for r in all_rows if r["comparison"] == "MISMATCH")
        missing_legacy = sum(1 for r in all_rows if r["comparison"] == "NO_LEGACY_VALUE")
        missing_pyevall = sum(1 for r in all_rows if r["comparison"] == "NO_PYEVALL_VALUE")

        html = [
            "<!doctype html>",
            "<html lang='en'>",
            "<head><meta charset='utf-8'><title>Clustering Metrics Comparison</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; background: #fffdf7; color: #333; margin: 20px; }",
            "h1 { color: #5f5b32; }",
            ".summary { background: #fff4cc; border: 1px solid #f0dea3; padding: 10px; border-radius: 8px; }",
            ".legend { margin-top: 12px; background: #f9f4dc; border: 1px solid #e8dcaa; padding: 10px; border-radius: 8px; }",
            "table { border-collapse: collapse; width: 100%; margin-top: 14px; background: #fffefb; }",
            "th { background: #fff0a8; border: 1px solid #e6d98f; padding: 8px; }",
            "td { border: 1px solid #ece2b7; padding: 7px; vertical-align: top; }",
            ".row-header { background: #fff8cc; font-weight: 600; }",
            ".match { background: #dff6e6; color: #1d6b35; font-weight: 700; }",
            ".mismatch { background: #ffe0e0; color: #a12727; font-weight: 700; }",
            ".no-legacy { background: #efe9ff; color: #5b3d92; font-weight: 700; }",
            ".unexpected { background: #ffe8d9; color: #8a4b1f; font-weight: 700; }",
            ".no-pyevall { background: #ffeaea; color: #993636; font-weight: 700; }",
            ".separator td { border-top: 3px double #8f8f8f; background: #fffdf4; height: 4px; padding: 0; }",
            ".error-cell { max-width: 520px; word-break: break-word; font-size: 12px; color: #6f1d1d; }",
            ".tabs { margin-top: 14px; }",
            ".tab-btn { background: #fff2bc; border: 1px solid #e6d98f; padding: 8px 12px; cursor: pointer; margin-right: 6px; border-radius: 6px; }",
            ".tab-btn.active { background: #ffe58e; font-weight: 700; }",
            ".tab-pane { display: none; }",
            ".tab-pane.active { display: block; }",
            "</style>",
            "<script>",
            "function openTab(tabId, btn){",
            "  document.querySelectorAll('.tab-pane').forEach(x=>x.classList.remove('active'));",
            "  document.querySelectorAll('.tab-btn').forEach(x=>x.classList.remove('active'));",
            "  document.getElementById(tabId).classList.add('active');",
            "  btn.classList.add('active');",
            "}",
            "</script>",
            "</head>",
            "<body>",
            "<h1>Clustering detailed comparison report (legacy vs pyevall)</h1>",
            "<div class='summary'>",
            f"<div><strong>Generated:</strong> {datetime.now().isoformat(timespec='seconds')}</div>",
            f"<div><strong>Total:</strong> {total} | <strong>MATCH:</strong> {matches} | <strong>MISMATCH:</strong> {mismatches} | <strong>NO_LEGACY_VALUE:</strong> {missing_legacy} | <strong>NO_PYEVALL_VALUE:</strong> {missing_pyevall}</div>",
            "</div>",
            "<div class='legend'><strong>Legend:</strong> MATCH = same values. MISMATCH = different values. NO_LEGACY_VALUE = legacy has '-'. NO_PYEVALL_VALUE = PyEvALL did not return value. UNEXPECTED_PYEVALL_VALUE = legacy has '-' but PyEvALL returned a value.</div>",
            "<div class='tabs'>",
        ]
        for i, (tab_id, tab_label, _) in enumerate(sections):
            active = " active" if i == 0 else ""
            html.append(f"<button class='tab-btn{active}' onclick=\"openTab('{tab_id}', this)\">{tab_label}</button>")
        html.append("</div>")
        for i, (tab_id, _, rows) in enumerate(sections):
            active = " active" if i == 0 else ""
            html.append(f"<div id='{tab_id}' class='tab-pane{active}'>")
            html.append("<table border='1' cellpadding='4' cellspacing='0'>")
            html.append("<thead><tr><th>metric</th><th>test_case</th><th>description</th><th>legacy_value</th><th>pyevall_value</th><th>abs_diff</th><th>status</th><th>comparison</th><th>pyevall_error</th></tr></thead>")
            html.append("<tbody>")
            previous_metric = None
            for row in rows:
                if previous_metric is not None and row["metric"] != previous_metric:
                    html.append("<tr class='separator'><td colspan='9'></td></tr>")
                previous_metric = row["metric"]
                comparison_class = {
                    "MATCH": "match",
                    "MISMATCH": "mismatch",
                    "NO_LEGACY_VALUE": "no-legacy",
                    "UNEXPECTED_PYEVALL_VALUE": "unexpected",
                    "NO_PYEVALL_VALUE": "no-pyevall",
                }.get(row["comparison"], "")
                html.append(
                    "<tr>"
                    f"<td class='row-header'>{row['metric']}</td>"
                    f"<td class='row-header'>{row['test_case']}</td>"
                    f"<td>{row['test_case_description']}</td>"
                    f"<td>{row['legacy_value']}</td>"
                    f"<td>{row['pyevall_value']}</td>"
                    f"<td>{row['abs_diff']}</td>"
                    f"<td>{row['status']}</td>"
                    f"<td class='{comparison_class}'>{row['comparison']}</td>"
                    f"<td class='error-cell'>{row['pyevall_error']}</td>"
                    "</tr>"
                )
            html.extend(["</tbody>", "</table>", "</div>"])
        html.extend(["</body>", "</html>"])
        try:
            self.REPORT_HTML.write_text("\n".join(html) + "\n", encoding="utf-8")
        except PermissionError as exc:
            raise AssertionError(
                f"ERROR_WRITING_HTML_REPORT: cannot write '{self.REPORT_HTML}'. "
                "Close any app locking the file and check write permissions."
            ) from exc

    # Casos V1:
    # Se usaran los archivos de test/resources/metrics/double_imp/clustering/V1_enrique.
    # El objetivo sera ejecutar metricas de clustering y comparar con su implementacion de referencia.
    def test_v1_enrique_compare_with_pyevall_and_generate_html(self, tmp_path):
        # Compare V1 clustering legacy results with PyEvALL and generate HTML.
        self._init_run_report_dir("V1_enrique")

        self._ensure_pyevall_json_from_tsv_like(self.V1_GOLD_SOURCE_FILE, self.V1_GOLD_PYEVALL_JSON)
        self._ensure_pyevall_json_from_tsv_like(self.V1_SYSA_SOURCE_FILE, self.V1_SYSA_PYEVALL_JSON)
        self._ensure_pyevall_json_from_tsv_like(self.V1_SYSB_SOURCE_FILE, self.V1_SYSB_PYEVALL_JSON)

        rows_sys_a = self._build_v1_comparison_rows(tmp_path, "ClusSysA.csv", self.V1_SYSA_PYEVALL_JSON)
        rows_sys_b = self._build_v1_comparison_rows(tmp_path, "ClusSysB.csv", self.V1_SYSB_PYEVALL_JSON)
        self._write_html_report(
            [
                ("tab_v1_sysa", "V1_enrique - SysA", rows_sys_a),
                ("tab_v1_sysb", "V1_enrique - SysB", rows_sys_b),
            ]
        )

        assert self.REPORT_HTML.exists()
        report_text = self.REPORT_HTML.read_text(encoding="utf-8")
        assert "V1_enrique - SysA" in report_text
        assert "V1_enrique - SysB" in report_text

    # Casos V2:
    # Se usaran los archivos de test/resources/metrics/double_imp/clustering/V2_enrique.
    # El objetivo sera ejecutar metricas de clustering y comparar con su implementacion de referencia.
    def test_v2_enrique_compare_with_pyevall_and_generate_html(self, tmp_path):
        # Compare V2 clustering legacy results with PyEvALL and generate HTML.
        self._init_run_report_dir("V2_enrique")

        self._ensure_pyevall_json_from_tsv_like(self.V1_GOLD_SOURCE_FILE, self.V1_GOLD_PYEVALL_JSON)
        self._ensure_pyevall_json_from_tsv_like(self.V1_SYSA_SOURCE_FILE, self.V1_SYSA_PYEVALL_JSON)
        self._ensure_pyevall_json_from_tsv_like(self.V1_SYSB_SOURCE_FILE, self.V1_SYSB_PYEVALL_JSON)
        self._ensure_pyevall_json_from_tsv_like(self.V2_GOLD_SOURCE_FILE, self.V2_GOLD_PYEVALL_JSON)
        self._ensure_pyevall_json_from_tsv_like(self.V2_SYS_SOURCE_FILE, self.V2_SYS_PYEVALL_JSON)

        rows_v1_a = self._build_v1_comparison_rows(tmp_path, "ClusSysA.csv", self.V1_SYSA_PYEVALL_JSON)
        rows_v1_b = self._build_v1_comparison_rows(tmp_path, "ClusSysB.csv", self.V1_SYSB_PYEVALL_JSON)
        rows_v2 = self._build_v2_comparison_rows(tmp_path, "SysEstressTestClustering.csv", self.V2_SYS_PYEVALL_JSON)

        self._write_html_report(
            [
                ("tab_v1_sysa", "V1_enrique - SysA", rows_v1_a),
                ("tab_v1_sysb", "V1_enrique - SysB", rows_v1_b),
                ("tab_v2_sys", "V2_enrique - SysEstressTestClustering", rows_v2),
            ]
        )

        assert self.REPORT_HTML.exists()
        report_text = self.REPORT_HTML.read_text(encoding="utf-8")
        assert "V1_enrique - SysA" in report_text
        assert "V1_enrique - SysB" in report_text
        assert "V2_enrique - SysEstressTestClustering" in report_text
