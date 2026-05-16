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
Format parsing/validation tests for PyEvALL.

This module reuses the legacy scenarios from test_old/testformats.py and extends
coverage with additional cases derived from README format definitions.

Covered checks:
1. Missing predictions file path.
2. Missing gold file path.
3. Empty predictions file.
4. Empty gold file.
5. Incorrect JSON in predictions.
6. Incorrect JSON in gold.
7. JSON schema violations in predictions.
8. JSON schema violations in gold.
9. Duplicated IDs in predictions.
10. Auto-detection TSV wrapper and successful evaluation.
11. Auto-detection CSV wrapper and successful evaluation.
12. Ranking duplicated score warning in gold.
13. Mixed value types inside one testcase (format error).
14. Different value types between gold and predictions (format error).
15. Valid clustering/multilabel list format in JSON.
16. Clustering duplicated tag in one item's list (new format error).
17. Clustering empty tag in one item's list (new format error).
18. Clustering list format parsed correctly from TSV wrapper.
19. Ranking files with value as string trigger not-valid-format precondition.
20. Ranking files with float in value fail JSON schema.
21. LeWiDi valid JSON evaluates without format errors.
22. LeWiDi with different classes in gold/pred remains format-valid.
23. LeWiDi with non-numeric probability fails schema.
24. LeWiDi metric over array value triggers not-valid-format precondition.
25. LeWiDi parsed from CSV wrapper stays valid for CrossEntropy.
26. Clustering array with non-string item fails schema.
27. Clustering array with nested array fails schema.
28. Malformed list in TSV wrapper triggers different-types-in-value error.
29. JSON missing required id fails schema.
30. JSON with additional property fails schema.
31. Duplicate IDs in gold are reported as format error.
32. No common test_case between files reports metric not-test-case-in-common.
33. Empty monolabel value is reported as warning and ignored.
34. Empty multilabel array is reported as warning and ignored.
35. Empty ranking value is reported as warning and ignored.
36. Kappa is not evaluated when predictions become empty after filtering empty values.
"""

from datetime import datetime
from html import escape
from pathlib import Path
import inspect

import pytest

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from pyevall.reports.reports import PyEvALLReport
from pyevall.utils.utils import PyEvALLUtils


@pytest.mark.format
class TestFormats:
    BASE = Path(__file__).resolve().parent / "resources" / "format"
    REPORT_DIR = Path(__file__).resolve().parent / "reports"
    REPORT_HTML = REPORT_DIR / "test_format_report.html"
    FORMAT_REPORT_ROWS = []

    @pytest.fixture(scope="class", autouse=True)
    def _format_html_report_fixture(self, request):
        # Collect format-test outcomes and write the HTML summary at class teardown.
        request.cls.FORMAT_REPORT_ROWS = []
        yield
        request.cls._write_format_html_report()

    def _evaluate(self, pred_path, gold_path, metrics=None):
        # Execute PyEvALL for a format fixture pair.
        metrics = metrics or []
        params = {PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE}
        report_object = PyEvALLEvaluation().evaluate(str(pred_path), str(gold_path), metrics, **params)
        self._append_format_report_row(pred_path, gold_path, metrics, report_object.report)
        return report_object.report

    def _file_errors(self, report, file_name):
        # Return file-level parser errors from a report.
        return report[PyEvALLReport.FILES_TAG][file_name][PyEvALLReport.ERRORS_TAG]

    def _metric(self, report, metric_name):
        # Return one metric section from a report.
        return report[PyEvALLReport.METRIC_TAG][metric_name]

    def _append_format_report_row(self, pred_path, gold_path, metrics, report):
        # Add one row to the format HTML report summary.
        test_name = next((frame.function for frame in inspect.stack() if frame.function.startswith("test_")), "unknown")
        outcome, outcome_class = self._format_report_outcome(report)
        self.FORMAT_REPORT_ROWS.append(
            {
                "test": test_name,
                "outcome": outcome,
                "outcome_class": outcome_class,
                "pred": str(pred_path),
                "gold": str(gold_path),
                "metrics": ", ".join(metrics) if metrics else "-",
                "files": self._summarize_files(report),
                "metric_status": self._summarize_metrics(report),
            }
        )

    @staticmethod
    def _summarize_files(report):
        # Summarize parsed file statuses and parser diagnostics.
        file_rows = []
        for file_name, file_report in report.get(PyEvALLReport.FILES_TAG, {}).items():
            errors = file_report.get(PyEvALLReport.ERRORS_TAG, {})
            if not errors:
                file_rows.append(f"{file_name}: OK")
                continue
            file_rows.append(f"{file_name}: {', '.join(errors.keys())}")
        return " | ".join(file_rows)

    @staticmethod
    def _summarize_metrics(report):
        # Summarize metric statuses and metric diagnostics.
        metric_rows = []
        for metric_name, metric_report in report.get(PyEvALLReport.METRIC_TAG, {}).items():
            status = metric_report.get(PyEvALLReport.STATUS_TAG, "-")
            issues = []
            if PyEvALLReport.ERRORS_TAG in metric_report:
                issues.extend(error.get(PyEvALLReport.NAME_TAG, str(error)) for error in metric_report[PyEvALLReport.ERRORS_TAG])
            if PyEvALLReport.PRECONDITIONS_TAG in metric_report:
                issues.extend(metric_report[PyEvALLReport.PRECONDITIONS_TAG].keys())
            suffix = f" ({', '.join(issues)})" if issues else ""
            metric_rows.append(f"{metric_name}: {status}{suffix}")
        return " | ".join(metric_rows) if metric_rows else "-"

    @staticmethod
    def _format_report_outcome(report):
        # Convert test success and expected-issue flags into a report label.
        has_file_issue = any(
            bool(file_report.get(PyEvALLReport.ERRORS_TAG, {}))
            for file_report in report.get(PyEvALLReport.FILES_TAG, {}).values()
        )
        has_metric_issue = any(
            metric_report.get(PyEvALLReport.STATUS_TAG) in [PyEvALLReport.FAIL, PyEvALLReport.WARNING]
            or bool(metric_report.get(PyEvALLReport.ERRORS_TAG, []))
            or bool(metric_report.get(PyEvALLReport.PRECONDITIONS_TAG, {}))
            for metric_report in report.get(PyEvALLReport.METRIC_TAG, {}).values()
        )
        if has_file_issue or has_metric_issue:
            return "EXPECTED ISSUE", "expected"
        return "OK", "ok"

    @classmethod
    def _write_format_html_report(cls):
        # Render collected format-test outcomes as an HTML report.
        cls.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        rows = cls.FORMAT_REPORT_ROWS
        html = [
            "<!doctype html>",
            "<html lang='en'>",
            "<head><meta charset='utf-8'><title>Format Tests Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; background: #f8fbff; color: #26313f; margin: 20px; }",
            "h1 { color: #24466f; }",
            ".summary { background: #eaf3ff; border: 1px solid #bdd7f5; padding: 10px; border-radius: 8px; margin-top: 10px; }",
            "table { border-collapse: collapse; width: 100%; margin-top: 14px; background: #ffffff; }",
            "th { background: #d7e9ff; border: 1px solid #bad2ed; padding: 8px; }",
            "td { border: 1px solid #d7e2ef; padding: 7px; vertical-align: top; font-size: 13px; }",
            ".test { background: #eef6ff; font-weight: 600; white-space: nowrap; }",
            ".details { max-width: 520px; word-break: break-word; }",
            ".ok { background: #dff6e6; color: #1d6b35; font-weight: 700; text-align: center; }",
            ".expected { background: #fff1cc; color: #7a5700; font-weight: 700; text-align: center; }",
            "</style>",
            "</head><body>",
            "<h1>Format tests report</h1>",
            "<div class='summary'>",
            f"<div><strong>Generated:</strong> {datetime.now().isoformat(timespec='seconds')}</div>",
            f"<div><strong>Evaluations:</strong> {len(rows)}</div>",
            "</div>",
            "<table><thead><tr><th>result</th><th>test</th><th>predictions</th><th>gold</th><th>metrics</th><th>file checks</th><th>metric checks</th></tr></thead><tbody>",
        ]
        for row in rows:
            html.append(
                "<tr>"
                f"<td class='{escape(row['outcome_class'])}'>{escape(row['outcome'])}</td>"
                f"<td class='test'>{escape(row['test'])}</td>"
                f"<td class='details'>{escape(row['pred'])}</td>"
                f"<td class='details'>{escape(row['gold'])}</td>"
                f"<td>{escape(row['metrics'])}</td>"
                f"<td class='details'>{escape(row['files'])}</td>"
                f"<td class='details'>{escape(row['metric_status'])}</td>"
                "</tr>"
            )
        html.extend(["</tbody></table>", "</body></html>"])
        cls.REPORT_HTML.write_text("\n".join(html) + "\n", encoding="utf-8")

    # -------------------------------------------------------------------------
    # Aqui empiezan pruebas de formato JSON
    # -------------------------------------------------------------------------

    @pytest.mark.json_format
    # Comprueba: error de archivo inexistente en predictions.
    def test_format_json_incorrect_url_prediction(self):
        # Check that an invalid prediction path is reported as a format error.
        gold = self.BASE / "json" / "GOLD_MONO.json"
        report = self._evaluate("asdf", gold)
        errors = self._file_errors(report, "asdf")
        assert PyEvALLReport.FORMAT_FILE_NOT_EXIST_ERROR in errors

    @pytest.mark.json_format
    # Comprueba: error de archivo inexistente en gold.
    def test_format_json_incorrect_url_gold(self):
        # Check that an invalid gold path is reported as a format error.
        report = self._evaluate("asdf", "sd")
        errors = self._file_errors(report, "sd")
        assert PyEvALLReport.FORMAT_FILE_NOT_EXIST_ERROR in errors

    @pytest.mark.json_format
    # Comprueba: deteccion de predictions vacio.
    def test_format_json_empty_prediction(self):
        # Check that an empty prediction file is reported as a format error.
        pred = self.BASE / "json" / "EMPTY"
        gold = self.BASE / "json" / "GOLD_MONO.json"
        report = self._evaluate(pred, gold)
        errors = self._file_errors(report, "EMPTY")
        assert PyEvALLReport.FORMAT_EMPTY_FILE_ERROR in errors

    @pytest.mark.json_format
    # Comprueba: deteccion de gold vacio.
    def test_format_json_empty_gold(self):
        # Check that an empty gold file is reported as a format error.
        pred = self.BASE / "json" / "GOLD_MONO.json"
        gold = self.BASE / "json" / "EMPTY"
        report = self._evaluate(pred, gold)
        errors = self._file_errors(report, "EMPTY")
        assert PyEvALLReport.FORMAT_EMPTY_FILE_ERROR in errors

    @pytest.mark.json_format
    # Comprueba: deteccion de JSON malformado en predictions.
    def test_format_json_incorrect_prediction(self):
        # Check that malformed prediction JSON is rejected.
        pred = self.BASE / "json" / "INCORRECT.json"
        gold = self.BASE / "json" / "GOLD_MONO.json"
        report = self._evaluate(pred, gold)
        errors = self._file_errors(report, "INCORRECT.json")
        assert PyEvALLReport.FORMAT_INCORRECT_JSON_ERROR in errors

    @pytest.mark.json_format
    # Comprueba: deteccion de JSON malformado en gold.
    def test_format_json_incorrect_gold(self):
        # Check that malformed gold JSON is rejected.
        pred = self.BASE / "json" / "GOLD_MONO.json"
        gold = self.BASE / "json" / "INCORRECT.json"
        report = self._evaluate(pred, gold)
        errors = self._file_errors(report, "INCORRECT.json")
        assert PyEvALLReport.FORMAT_INCORRECT_JSON_ERROR in errors

    @pytest.mark.json_format
    # Comprueba: deteccion de schema invalido en predictions.
    def test_format_json_incorrect_schema_prediction(self):
        # Check that prediction JSON schema errors are reported.
        pred = self.BASE / "json" / "SCHEMA_INCORRECT.json"
        gold = self.BASE / "json" / "GOLD_MONO.json"
        report = self._evaluate(pred, gold)
        errors = self._file_errors(report, "SCHEMA_INCORRECT.json")
        assert PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR in errors

    @pytest.mark.json_format
    # Comprueba: deteccion de schema invalido en gold.
    def test_format_json_incorrect_schema_gold(self):
        # Check that gold JSON schema errors are reported.
        pred = self.BASE / "json" / "GOLD_MONO.json"
        gold = self.BASE / "json" / "SCHEMA_INCORRECT.json"
        report = self._evaluate(pred, gold)
        errors = self._file_errors(report, "SCHEMA_INCORRECT.json")
        assert PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR in errors

    @pytest.mark.json_format
    # Comprueba: deteccion de IDs duplicados en predictions.
    def test_format_json_duplicates_ids_prediction(self):
        # Check that duplicated prediction ids are reported.
        pred = self.BASE / "json" / "SYS_DUPLICATE_IDS.json"
        gold = self.BASE / "json" / "GOLD_MONO.json"
        report = self._evaluate(pred, gold)
        errors = self._file_errors(report, "SYS_DUPLICATE_IDS.json")
        assert PyEvALLReport.FORMAT_IDS_REPEATED_ROW_ERROR in errors

    # -------------------------------------------------------------------------
    # Aqui empiezan pruebas de wrappers (TSV/CSV)
    # -------------------------------------------------------------------------

    @pytest.mark.wrappers
    # Comprueba: wrapper TSV identificado y evaluacion correcta.
    def test_format_json_vs_tsv_correct(self):
        # Check that equivalent JSON and TSV inputs produce the same score.
        pred = self.BASE / "json_vs_tsv" / "SYS1.txt"
        gold = self.BASE / "json_vs_tsv" / "GOLD1.txt"
        report = self._evaluate(pred, gold, [MetricFactory.Accuracy.value])

        file_errors = self._file_errors(report, "SYS1.txt")
        assert PyEvALLReport.FORMAT_TSV_FORMAT_IDENTIFIED_WARNING in file_errors

        acc = report[PyEvALLReport.METRIC_TAG][MetricFactory.Accuracy.value]
        assert acc[PyEvALLReport.RESULTS_TAG][PyEvALLReport.AVERAGE_PER_TC_TAG] == 1.0

    @pytest.mark.wrappers
    # Comprueba: wrapper CSV identificado y evaluacion correcta.
    def test_format_json_vs_csv_correct(self):
        # Check that equivalent JSON and CSV inputs produce the same score.
        pred = self.BASE / "csv_vs_json" / "SYS1.csv"
        gold = self.BASE / "json_vs_tsv" / "GOLD1.txt"
        report = self._evaluate(pred, gold, [MetricFactory.Accuracy.value])

        file_errors = self._file_errors(report, "SYS1.csv")
        assert PyEvALLReport.FORMAT_CSV_FORMAT_IDENTIFIED_WARNING in file_errors

        acc = report[PyEvALLReport.METRIC_TAG][MetricFactory.Accuracy.value]
        assert acc[PyEvALLReport.RESULTS_TAG][PyEvALLReport.AVERAGE_PER_TC_TAG] == 1.0

    # -------------------------------------------------------------------------
    # Aqui empiezan pruebas de consistencia de tipos
    # -------------------------------------------------------------------------

    @pytest.mark.consistency
    # Comprueba: warning de valores duplicados en ranking gold.
    def test_format_ranking_duplicated_value(self):
        # Check that duplicated ranking positions trigger the ranking precondition.
        pred = self.BASE / "duplicated_values" / "SYS1.txt"
        gold = self.BASE / "duplicated_values" / "GOLD1.txt"
        report = self._evaluate(pred, gold, [MetricFactory.PrecisionAtK.value])

        metric = report[PyEvALLReport.METRIC_TAG][MetricFactory.PrecisionAtK.value]
        assert PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING in metric[PyEvALLReport.PRECONDITIONS_TAG]

    @pytest.mark.consistency
    # Comprueba: error por tipos mezclados de value dentro de un test_case.
    def test_format_mixed_value_types_in_one_testcase(self):
        # Check that mixed value types inside one test case are rejected.
        pred = self.BASE / "type_consistency" / "GOLD_MONO_SMALL.json"
        gold = self.BASE / "type_consistency" / "GOLD_MIXED_TYPES.json"
        report = self._evaluate(pred, gold)

        errors = self._file_errors(report, "GOLD_MIXED_TYPES.json")
        assert PyEvALLReport.FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR in errors

    @pytest.mark.consistency
    # Comprueba: error por tipos distintos de value entre gold y pred.
    def test_format_different_value_types_gold_and_pred(self):
        # Check that incompatible gold/prediction value types are rejected.
        pred = self.BASE / "type_consistency" / "SYS_MULTI_SMALL.json"
        gold = self.BASE / "type_consistency" / "GOLD_MONO_SMALL.json"
        report = self._evaluate(pred, gold)

        errors = self._file_errors(report, "SYS_MULTI_SMALL.json")
        assert PyEvALLReport.FORMAT_DIFFERENT_TYPES_IN_VALUE_GOLD_AND_PRED in errors

    @pytest.mark.consistency
    # Comprueba: deteccion de IDs duplicados en gold.
    def test_format_duplicate_ids_gold(self):
        # Check that duplicated gold ids stop evaluation.
        pred = self.BASE / "type_consistency" / "GOLD_MONO_SMALL.json"
        gold = self.BASE / "json" / "GOLD_DUPLICATE_IDS.json"
        report = self._evaluate(pred, gold)

        errors = self._file_errors(report, "GOLD_DUPLICATE_IDS.json")
        assert PyEvALLReport.FORMAT_IDS_REPEATED_ROW_ERROR in errors

    # -------------------------------------------------------------------------
    # Aqui empiezan pruebas de clustering (value como lista)
    # -------------------------------------------------------------------------

    @pytest.mark.clustering
    # Comprueba: clustering JSON valido sin errores de formato.
    def test_format_clustering_json_valid(self):
        # Check that valid clustering JSON executes successfully.
        pred = self.BASE / "clustering" / "valid" / "SYS_CLUSTER_VALID.json"
        gold = self.BASE / "clustering" / "valid" / "GOLD_CLUSTER_VALID.json"
        report = self._evaluate(pred, gold)

        pred_errors = self._file_errors(report, "SYS_CLUSTER_VALID.json")
        gold_errors = self._file_errors(report, "GOLD_CLUSTER_VALID.json")
        assert len(pred_errors) == 0
        assert len(gold_errors) == 0

    @pytest.mark.clustering
    # Comprueba: error por etiqueta de cluster duplicada en una lista value.
    def test_format_clustering_duplicate_tag_in_value(self):
        # Check that duplicated cluster tags in one value are reported.
        pred = self.BASE / "clustering" / "errors" / "SYS_CLUSTER_DUPLICATE_TAG.json"
        gold = self.BASE / "clustering" / "errors" / "GOLD_CLUSTER_SINGLE.json"
        report = self._evaluate(pred, gold)

        pred_errors = self._file_errors(report, "SYS_CLUSTER_DUPLICATE_TAG.json")
        assert PyEvALLReport.FORMAT_CLUSTER_DUPLICATED_TAG_IN_VALUE_ROW_ERROR in pred_errors

    @pytest.mark.clustering
    # Comprueba: error por etiqueta de cluster vacia en una lista value.
    def test_format_clustering_empty_tag_in_value(self):
        # Check that empty cluster tags in one value are reported.
        pred = self.BASE / "clustering" / "errors" / "SYS_CLUSTER_EMPTY_TAG.json"
        gold = self.BASE / "clustering" / "errors" / "GOLD_CLUSTER_SINGLE.json"
        report = self._evaluate(pred, gold)

        pred_errors = self._file_errors(report, "SYS_CLUSTER_EMPTY_TAG.json")
        assert PyEvALLReport.FORMAT_CLUSTER_EMPTY_TAG_IN_VALUE_ROW_ERROR in pred_errors

    @pytest.mark.clustering
    @pytest.mark.wrappers
    # Comprueba: parseo clustering por wrapper TSV sin errores de etiquetas.
    def test_format_clustering_tsv_wrapper(self):
        # Check that clustering TSV wrappers are parsed correctly.
        pred = self.BASE / "clustering" / "tsv" / "SYS_CLUSTER_TSV.txt"
        gold = self.BASE / "clustering" / "tsv" / "GOLD_CLUSTER_TSV.json"
        report = self._evaluate(pred, gold)

        pred_errors = self._file_errors(report, "SYS_CLUSTER_TSV.txt")
        assert PyEvALLReport.FORMAT_TSV_FORMAT_IDENTIFIED_WARNING in pred_errors
        assert PyEvALLReport.FORMAT_CLUSTER_EMPTY_TAG_IN_VALUE_ROW_ERROR not in pred_errors
        assert PyEvALLReport.FORMAT_CLUSTER_DUPLICATED_TAG_IN_VALUE_ROW_ERROR not in pred_errors

    @pytest.mark.clustering
    # Comprueba: schema invalido cuando value contiene item no string.
    def test_format_clustering_non_string_item_schema(self):
        # Check that non-string clustering array items fail schema validation.
        pred = self.BASE / "clustering" / "errors" / "SYS_CLUSTER_NON_STRING_TAG.json"
        gold = self.BASE / "clustering" / "errors" / "GOLD_CLUSTER_SINGLE.json"
        report = self._evaluate(pred, gold)

        pred_errors = self._file_errors(report, "SYS_CLUSTER_NON_STRING_TAG.json")
        assert PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR in pred_errors

    @pytest.mark.clustering
    # Comprueba: schema invalido cuando value contiene arrays anidados.
    def test_format_clustering_nested_array_schema(self):
        # Check that nested clustering arrays fail schema validation.
        pred = self.BASE / "clustering" / "errors" / "SYS_CLUSTER_NESTED_ARRAY.json"
        gold = self.BASE / "clustering" / "errors" / "GOLD_CLUSTER_SINGLE.json"
        report = self._evaluate(pred, gold)

        pred_errors = self._file_errors(report, "SYS_CLUSTER_NESTED_ARRAY.json")
        assert PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR in pred_errors

    @pytest.mark.clustering
    @pytest.mark.wrappers
    # Comprueba: error por tipos distintos gold/pred tras lista TSV malformada.
    def test_format_clustering_tsv_invalid_list_mixed_types(self):
        # Check that invalid clustering lists converted from TSV are rejected.
        pred = self.BASE / "clustering" / "tsv" / "SYS_CLUSTER_TSV_INVALID_LIST.txt"
        gold = self.BASE / "clustering" / "tsv" / "GOLD_CLUSTER_TSV.json"
        report = self._evaluate(pred, gold)

        pred_errors = self._file_errors(report, "SYS_CLUSTER_TSV_INVALID_LIST.txt")
        assert PyEvALLReport.FORMAT_DIFFERENT_TYPES_IN_VALUE_GOLD_AND_PRED in pred_errors

    # -------------------------------------------------------------------------
    # Aqui empiezan pruebas de ranking
    # -------------------------------------------------------------------------

    @pytest.mark.consistency
    # Comprueba: precondicion de formato invalido en metrica ranking con values string.
    def test_format_ranking_string_values_not_valid_for_ranking_metric(self):
        # Check that string values fail ranking metric preconditions.
        pred = self.BASE / "ranking_errors" / "SYS_RANKING_STRING.json"
        gold = self.BASE / "ranking_errors" / "GOLD_RANKING_STRING.json"
        report = self._evaluate(pred, gold, [MetricFactory.PrecisionAtK.value])

        metric = self._metric(report, MetricFactory.PrecisionAtK.value)
        assert PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION in metric[PyEvALLReport.PRECONDITIONS_TAG]

    @pytest.mark.consistency
    # Comprueba: schema invalido cuando ranking usa float.
    def test_format_ranking_float_schema_error(self):
        # Check that float ranking values fail schema validation.
        pred = self.BASE / "ranking_errors" / "SYS_RANKING_FLOAT_SCHEMA.json"
        gold = self.BASE / "ranking_errors" / "GOLD_RANKING_FLOAT_SCHEMA.json"
        report = self._evaluate(pred, gold)

        errors = self._file_errors(report, "GOLD_RANKING_FLOAT_SCHEMA.json")
        assert PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR in errors

    # -------------------------------------------------------------------------
    # Aqui empiezan pruebas de LeWiDi
    # -------------------------------------------------------------------------

    @pytest.mark.consistency
    # Comprueba: LeWiDi valido para CrossEntropy sin errores de formato.
    def test_format_lewedi_valid_cross_entropy(self):
        # Check that valid LeWiDi distributions execute CrossEntropy.
        pred = self.BASE / "lewedi" / "SYS_LEWEDI_VALID.json"
        gold = self.BASE / "lewedi" / "GOLD_LEWEDI_VALID.json"
        report = self._evaluate(pred, gold, [MetricFactory.CrossEntropy.value])

        assert len(self._file_errors(report, "SYS_LEWEDI_VALID.json")) == 0
        assert len(self._file_errors(report, "GOLD_LEWEDI_VALID.json")) == 0
        metric = self._metric(report, MetricFactory.CrossEntropy.value)
        assert PyEvALLReport.RESULTS_TAG in metric

    @pytest.mark.consistency
    # Comprueba: LeWiDi valido aunque clases difieran entre gold y pred.
    def test_format_lewedi_different_classes_gold_pred_valid_format(self):
        # Check that LeWiDi files with different class keys remain valid format.
        pred = self.BASE / "lewedi" / "SYS_LEWEDI_DIFF_CLASSES.json"
        gold = self.BASE / "lewedi" / "GOLD_LEWEDI_DIFF_CLASSES.json"
        report = self._evaluate(pred, gold, [MetricFactory.CrossEntropy.value])

        assert len(self._file_errors(report, "SYS_LEWEDI_DIFF_CLASSES.json")) == 0
        assert len(self._file_errors(report, "GOLD_LEWEDI_DIFF_CLASSES.json")) == 0
        metric = self._metric(report, MetricFactory.CrossEntropy.value)
        assert metric[PyEvALLReport.STATUS_TAG] in [PyEvALLReport.OK, PyEvALLReport.WARNING]

    @pytest.mark.consistency
    # Comprueba: schema invalido por probabilidad no numerica en LeWiDi.
    def test_format_lewedi_non_numeric_probability_schema_error(self):
        # Check that non-numeric LeWiDi probabilities fail schema validation.
        pred = self.BASE / "lewedi" / "SYS_LEWEDI_VALUE_STRING.json"
        gold = self.BASE / "lewedi" / "GOLD_LEWEDI_VALUE_STRING.json"
        report = self._evaluate(pred, gold)

        errors = self._file_errors(report, "GOLD_LEWEDI_VALUE_STRING.json")
        assert PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR in errors

    @pytest.mark.consistency
    # Comprueba: precondicion invalida para metrica LeWiDi cuando value es array.
    def test_format_lewedi_array_value_not_valid_for_lewedi_metric(self):
        # Check that array values fail LeWiDi metric preconditions.
        pred = self.BASE / "lewedi" / "SYS_LEWEDI_ARRAY.json"
        gold = self.BASE / "lewedi" / "GOLD_LEWEDI_ARRAY.json"
        report = self._evaluate(pred, gold, [MetricFactory.CrossEntropy.value])

        metric = self._metric(report, MetricFactory.CrossEntropy.value)
        assert PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION in metric[PyEvALLReport.PRECONDITIONS_TAG]

    @pytest.mark.wrappers
    @pytest.mark.consistency
    # Comprueba: wrapper CSV para LeWiDi valido y usable en CrossEntropy.
    def test_format_lewedi_csv_wrapper_valid(self):
        # Check that LeWiDi CSV wrappers are parsed correctly.
        pred = self.BASE / "lewedi" / "SYS_LEWEDI_CSV.csv"
        gold = self.BASE / "lewedi" / "GOLD_LEWEDI_VALID.json"
        report = self._evaluate(pred, gold, [MetricFactory.CrossEntropy.value])

        pred_errors = self._file_errors(report, "SYS_LEWEDI_CSV.csv")
        assert PyEvALLReport.FORMAT_CSV_FORMAT_IDENTIFIED_WARNING in pred_errors
        metric = self._metric(report, MetricFactory.CrossEntropy.value)
        assert metric[PyEvALLReport.STATUS_TAG] in [PyEvALLReport.OK, PyEvALLReport.WARNING]

    # -------------------------------------------------------------------------
    # Aqui empiezan pruebas de schema general y estructura
    # -------------------------------------------------------------------------

    @pytest.mark.json_format
    # Comprueba: schema invalido cuando falta campo requerido id.
    def test_format_schema_missing_required_id(self):
        # Check that missing required id fields fail schema validation.
        pred = self.BASE / "json" / "SYS_MONO.json"
        gold = self.BASE / "json" / "SCHEMA_MISSING_ID.json"
        report = self._evaluate(pred, gold)

        errors = self._file_errors(report, "SCHEMA_MISSING_ID.json")
        assert PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR in errors

    @pytest.mark.json_format
    # Comprueba: schema invalido con propiedad extra no permitida.
    def test_format_schema_additional_property_not_allowed(self):
        # Check that unexpected JSON properties fail schema validation.
        pred = self.BASE / "json" / "SYS_MONO.json"
        gold = self.BASE / "json" / "SCHEMA_EXTRA_PROPERTY.json"
        report = self._evaluate(pred, gold)

        errors = self._file_errors(report, "SCHEMA_EXTRA_PROPERTY.json")
        assert PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR in errors

    @pytest.mark.consistency
    # Comprueba: error de metrica cuando no hay test_case en comun.
    def test_format_no_common_testcase_reports_metric_error(self):
        # Check that files without common test cases report a metric error.
        pred = self.BASE / "json" / "SYS_TC_ONLY.json"
        gold = self.BASE / "json" / "GOLD_TC_ONLY.json"
        report = self._evaluate(pred, gold, [MetricFactory.Accuracy.value])

        metric = self._metric(report, MetricFactory.Accuracy.value)
        assert PyEvALLReport.ERRORS_TAG in metric
        assert metric[PyEvALLReport.ERRORS_TAG][0][PyEvALLReport.NAME_TAG] == PyEvALLReport.METRIC_NOT_TEST_CASE_IN_COMMON_ERROR

    # -------------------------------------------------------------------------
    # Aqui empiezan pruebas de valores vacios ignorados
    # -------------------------------------------------------------------------

    @pytest.mark.json_format
    @pytest.mark.consistency
    # Comprueba: value "" en gold se reporta como warning y se ignora junto a su prediccion.
    def test_format_empty_string_gold_value_warning_and_ignored(self):
        # Check that empty gold string values are warned and ignored.
        pred = self.BASE / "empty_values" / "SYS_MONO_EMPTY_GOLD_PAIR.json"
        gold = self.BASE / "empty_values" / "GOLD_MONO_EMPTY_VALUE.json"
        report = self._evaluate(pred, gold, [MetricFactory.SystemPrecision.value])

        gold_errors = self._file_errors(report, "GOLD_MONO_EMPTY_VALUE.json")
        assert PyEvALLReport.FORMAT_EMPTY_VALUE_ROW_WARNING in gold_errors

        metric = self._metric(report, MetricFactory.SystemPrecision.value)
        assert metric[PyEvALLReport.RESULTS_TAG][PyEvALLReport.AVERAGE_PER_TC_TAG] == 1.0

    @pytest.mark.json_format
    @pytest.mark.consistency
    # Comprueba: value [] en multilabel se reporta como warning y se ignora.
    def test_format_empty_array_multilabel_value_warning_and_ignored(self):
        # Check that empty multilabel arrays are warned and ignored.
        pred = self.BASE / "empty_values" / "SYS_MULTI_EMPTY_GOLD_PAIR.json"
        gold = self.BASE / "empty_values" / "GOLD_MULTI_EMPTY_ARRAY.json"
        report = self._evaluate(pred, gold, [MetricFactory.Precision.value])

        gold_errors = self._file_errors(report, "GOLD_MULTI_EMPTY_ARRAY.json")
        assert PyEvALLReport.FORMAT_EMPTY_VALUE_ROW_WARNING in gold_errors

        metric = self._metric(report, MetricFactory.Precision.value)
        assert metric[PyEvALLReport.RESULTS_TAG][PyEvALLReport.AVERAGE_PER_TC_TAG] == 1.0

    @pytest.mark.json_format
    @pytest.mark.consistency
    # Comprueba: value "" en ranking se reporta como warning y se ignora.
    def test_format_empty_string_ranking_value_warning_and_ignored(self):
        # Check that empty ranking values are warned and ignored.
        pred = self.BASE / "empty_values" / "SYS_RANKING_EMPTY_GOLD_PAIR.json"
        gold = self.BASE / "empty_values" / "GOLD_RANKING_EMPTY_VALUE.json"
        report = self._evaluate(pred, gold, [MetricFactory.PrecisionAtK.value])

        gold_errors = self._file_errors(report, "GOLD_RANKING_EMPTY_VALUE.json")
        assert PyEvALLReport.FORMAT_EMPTY_VALUE_ROW_WARNING in gold_errors

        metric = self._metric(report, MetricFactory.PrecisionAtK.value)
        assert metric[PyEvALLReport.RESULTS_TAG][PyEvALLReport.AVERAGE_PER_TC_TAG] == 0.1

    @pytest.mark.json_format
    @pytest.mark.consistency
    # Comprueba: Kappa no se evalua si el system queda sin etiquetas validas tras filtrar vacios.
    def test_format_kappa_empty_predictions_after_empty_value_filtering(self):
        # Check that Kappa fails when filtering leaves no valid predictions.
        pred = self.BASE / "empty_values" / "SYS_MONO_ALL_EMPTY.json"
        gold = self.BASE / "empty_values" / "GOLD_MONO_KAPPA_EMPTY_PRED.json"
        report = self._evaluate(pred, gold, [MetricFactory.Kappa.value])

        pred_errors = self._file_errors(report, "SYS_MONO_ALL_EMPTY.json")
        assert PyEvALLReport.FORMAT_EMPTY_VALUE_ROW_WARNING in pred_errors

        metric = self._metric(report, MetricFactory.Kappa.value)
        assert PyEvALLReport.METRIC_PRECONDITION_EMPTY_PREDICTIONS_ERROR in metric[PyEvALLReport.PRECONDITIONS_TAG]
