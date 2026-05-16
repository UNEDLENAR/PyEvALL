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
Manual debug runner for clustering metrics.

This module is intentionally independent from pytest collection.
Use `python test/manual_debug/clustering_debug_main.py --help`.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils


class ClusteringDebugRunner:
    BASE_DIR = Path(__file__).resolve().parents[1] / "resources" / "metrics" / "double_imp" / "clustering" / "V0_old_evall"
    GOLD_SOURCE = BASE_DIR / "Pyevall_OLD_EVALL_gold.json"
    PRED_SOURCE = BASE_DIR / "Pyevall_OLD_EVALL_preditions.json"
    TMP_DIR = Path(__file__).resolve().parent / "tmp" / "clustering_debug"

    METRIC_FMEASURE_P_INV = "FMeasurePurityInversePurity"

    def _load_json(self, path):
        # Load a JSON fixture file.
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path, data):
        # Write a JSON fixture file with stable formatting.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def prepare_t11_files(self):
        # Prepare derived clustering fixtures for the T11 debug scenario.
        gold = self._load_json(self.GOLD_SOURCE)
        pred = self._load_json(self.PRED_SOURCE)

        gold_t11 = [row for row in gold if row.get("test_case") == "t11"]
        pred_t11 = [row for row in pred if row.get("test_case") == "t11"]

        if not gold_t11:
            raise RuntimeError("No gold rows found for t11.")
        if not pred_t11:
            raise RuntimeError("No prediction rows found for t11.")

        gold_path = self.TMP_DIR / "gold_t11.json"
        pred_path = self.TMP_DIR / "pred_t11.json"
        self._write_json(gold_path, gold_t11)
        self._write_json(pred_path, pred_t11)
        return gold_path, pred_path

    def run_fmeasure_t11(self, alfa_param=0.5):
        # Run the T11 F-measure debug scenario.
        gold_path, pred_path = self.prepare_t11_files()

        params = {
            getattr(PyEvALLUtils, "PARAM_REPORT", "report"): getattr(
                PyEvALLUtils, "PARAM_OPTION_REPORT_EMBEDDED", "embedded"
            ),
            "metric_params": {self.METRIC_FMEASURE_P_INV: {"alfa_param": alfa_param}},
        }
        if hasattr(PyEvALLUtils, "PARAM_LOG_LEVEL") and hasattr(PyEvALLUtils, "PARAM_OPTION_LOG_LEVEL_NONE"):
            params[PyEvALLUtils.PARAM_LOG_LEVEL] = PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE

        report_obj = PyEvALLEvaluation().evaluate(
            str(pred_path),
            str(gold_path),
            [self.METRIC_FMEASURE_P_INV],
            **params,
        )
        report = report_obj.report

        metric_report = report.get("metrics", {}).get(self.METRIC_FMEASURE_P_INV, {})
        preconditions = metric_report.get("preconditions", {})

        report_path = self.TMP_DIR / "report_t11_fmeasure_purity_inverse_purity.json"
        preconditions_path = self.TMP_DIR / "preconditions_t11_fmeasure_purity_inverse_purity.json"

        self._write_json(report_path, report)
        self._write_json(preconditions_path, preconditions)

        print("Debug execution completed.")
        print(f"Gold file: {gold_path}")
        print(f"Pred file: {pred_path}")
        print(f"Report file: {report_path}")
        print(f"Preconditions file: {preconditions_path}")

        results = metric_report.get("results", {})
        print("Metric status:", metric_report.get("status"))
        print("Average per test case:", results.get("average_per_test_case"))

    def run_method(self, method_name):
        # Dispatch a manual debug command to the selected scenario.
        methods = {
            "prepare_t11_files": self.prepare_t11_files,
            "run_fmeasure_t11": self.run_fmeasure_t11,
        }
        if method_name not in methods:
            raise ValueError(f"Unknown method '{method_name}'. Available: {', '.join(sorted(methods.keys()))}")
        return methods[method_name]()


def build_parser():
    # Build the command-line parser for manual debug execution.
    parser = argparse.ArgumentParser(description="Manual debug runner for clustering metrics.")
    parser.add_argument(
        "--method",
        default="run_fmeasure_t11",
        choices=["prepare_t11_files", "run_fmeasure_t11"],
        help="Method to execute.",
    )
    parser.add_argument(
        "--alfa",
        type=float,
        default=0.5,
        help="alfa_param for FMeasurePurityInversePurity.",
    )
    return parser


def main():
    # Parse command-line arguments and execute the selected debug scenario.
    args = build_parser().parse_args()
    runner = ClusteringDebugRunner()
    if args.method == "run_fmeasure_t11":
        runner.run_fmeasure_t11(alfa_param=args.alfa)
    else:
        gold_path, pred_path = runner.prepare_t11_files()
        print("Files prepared.")
        print(f"Gold file: {gold_path}")
        print(f"Pred file: {pred_path}")


if __name__ == "__main__":
    main()
