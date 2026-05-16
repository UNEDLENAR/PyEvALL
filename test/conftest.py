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

from datetime import datetime
from pathlib import Path


REPORT_TARGETS = {
    "test_formats.py": "test_formats_report.md",
    "test_clustering_metrics.py": "test_clustering_metrics_execution_report.md",
    "test_classification_metrics.py": "test_classification_metrics_execution_report.md",
    "test_ranking_metrics.py": "test_ranking_metrics_execution_report.md",
}
EXEC_ROWS = {k: [] for k in REPORT_TARGETS}


def _module_key_from_nodeid(nodeid):
    # Extract the test module file name from a pytest node id.
    file_part = nodeid.split("::", 1)[0].replace("\\", "/")
    return file_part.split("/")[-1]


def _test_name_from_nodeid(nodeid):
    # Extract the test name portion from a pytest node id.
    parts = nodeid.split("::")
    return "::".join(parts[1:]) if len(parts) > 1 else nodeid


def _metric_group_from_test_name(test_name):
    # Group clustering tests by metric for readable execution reports.
    if "test_clustering_metric_" not in test_name:
        return "other"
    suffix = test_name.split("test_clustering_metric_", 1)[1]
    suffix = suffix.split("_against_", 1)[0]
    suffix = suffix.split("_derived_", 1)[0]
    suffix = suffix.split("_executes", 1)[0]
    return suffix


def pytest_configure(config):
    # Reset collected execution rows at the beginning of a pytest run.
    for key in REPORT_TARGETS:
        EXEC_ROWS[key] = []


def pytest_runtest_logreport(report):
    # Collect the outcome of each executed test call.
    if report.when != "call":
        return
    module_key = _module_key_from_nodeid(report.nodeid)
    if module_key not in REPORT_TARGETS:
        return

    outcome = "PASS" if report.passed else "FAIL" if report.failed else "SKIP"
    EXEC_ROWS[module_key].append(
        {
            "test": _test_name_from_nodeid(report.nodeid),
            "outcome": outcome,
        }
    )


def pytest_sessionfinish(session, exitstatus):
    # Write compact Markdown execution reports after the pytest session.
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    for module_key, file_name in REPORT_TARGETS.items():
        rows = EXEC_ROWS.get(module_key, [])
        if not rows:
            continue

        if module_key == "test_clustering_metrics.py":
            rows = sorted(rows, key=lambda x: (_metric_group_from_test_name(x["test"]), x["test"]))
        else:
            rows = sorted(rows, key=lambda x: x["test"])
        total = len(rows)
        passed = sum(1 for r in rows if r["outcome"] == "PASS")
        failed = sum(1 for r in rows if r["outcome"] == "FAIL")
        skipped = sum(1 for r in rows if r["outcome"] == "SKIP")

        lines = []
        lines.append(f"# Pytest execution report: {module_key}")
        lines.append("")
        lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"- Total tests: {total}")
        lines.append(f"- PASS: {passed}")
        lines.append(f"- FAIL: {failed}")
        lines.append(f"- SKIP: {skipped}")
        lines.append("")
        lines.append("| test | outcome |")
        lines.append("|---|---|")
        previous_metric = None
        for row in rows:
            if module_key == "test_clustering_metrics.py":
                current_metric = _metric_group_from_test_name(row["test"])
                if previous_metric is not None and current_metric != previous_metric:
                    lines.append("| **====================** |  |")
                previous_metric = current_metric
            lines.append(f"| {row['test']} | {row['outcome']} |")

        target_path = report_dir / file_name
        try:
            target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except PermissionError:
            fallback_name = f"{Path(file_name).stem}_fallback.md"
            try:
                (report_dir / fallback_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
            except PermissionError:
                pass
