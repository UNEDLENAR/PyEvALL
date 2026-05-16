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

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from pyevall.reports.reports import PyEvALLReport
from pyevall.utils.utils import PyEvALLUtils


class TestMetricParamsWarnings:
    RESOURCES_DIR = Path(__file__).resolve().parent / "resources" / "format" / "type_consistency"
    PRED = RESOURCES_DIR / "GOLD_MONO_SMALL.json"
    GOLD = RESOURCES_DIR / "GOLD_MONO_SMALL.json"

    def _evaluate(self, metrics, **params):
        params.setdefault(PyEvALLUtils.PARAM_LOG_LEVEL, PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE)
        return PyEvALLEvaluation().evaluate(str(self.PRED), str(self.GOLD), metrics, **params).report

    def test_global_metric_params_not_dict_adds_warning_to_all_valid_metrics(self):
        metrics = [MetricFactory.Accuracy.value, MetricFactory.Precision.value]

        report = self._evaluate(metrics, metric_params="invalid")

        for metric_name in metrics:
            metric = report[PyEvALLReport.METRIC_TAG][metric_name]
            assert metric[PyEvALLReport.STATUS_TAG] == PyEvALLReport.WARNING
            assert metric[PyEvALLReport.WARNINGS_TAG][0][PyEvALLReport.NAME_TAG] == PyEvALLReport.METRIC_PARAMS_NOT_DICT_WARNING
            assert PyEvALLReport.RESULTS_TAG in metric

    def test_metric_specific_params_not_dict_adds_warning_only_to_that_metric(self):
        report = self._evaluate(
            [MetricFactory.Accuracy.value, MetricFactory.Precision.value],
            metric_params={MetricFactory.Accuracy.value: "invalid"},
        )

        accuracy = report[PyEvALLReport.METRIC_TAG][MetricFactory.Accuracy.value]
        precision = report[PyEvALLReport.METRIC_TAG][MetricFactory.Precision.value]

        assert accuracy[PyEvALLReport.STATUS_TAG] == PyEvALLReport.WARNING
        assert accuracy[PyEvALLReport.WARNINGS_TAG][0][PyEvALLReport.NAME_TAG] == PyEvALLReport.METRIC_PARAMS_FOR_METRIC_NOT_DICT_WARNING
        assert PyEvALLReport.WARNINGS_TAG not in precision

    def test_invalid_metric_parameter_adds_warning_inside_metric(self):
        report = self._evaluate(
            [MetricFactory.FMeasure.value, MetricFactory.Precision.value],
            metric_params={MetricFactory.FMeasure.value: {"alfa_param": "invalid"}},
        )

        fmeasure = report[PyEvALLReport.METRIC_TAG][MetricFactory.FMeasure.value]
        precision = report[PyEvALLReport.METRIC_TAG][MetricFactory.Precision.value]

        assert fmeasure[PyEvALLReport.STATUS_TAG] == PyEvALLReport.WARNING
        assert fmeasure[PyEvALLReport.WARNINGS_TAG][0][PyEvALLReport.NAME_TAG] == PyEvALLReport.METRIC_PARAM_ALFA_NOT_VALID_WARNING
        assert PyEvALLReport.WARNINGS_TAG not in precision

    def test_embedded_report_expands_metric_params_warning_description(self):
        report = self._evaluate(
            [MetricFactory.Accuracy.value],
            metric_params="invalid",
            report=PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED,
        )

        warning = report[PyEvALLReport.METRIC_TAG][MetricFactory.Accuracy.value][PyEvALLReport.WARNINGS_TAG][0]

        assert warning[PyEvALLReport.DESCRIPTION_TAG] != PyEvALLReport.EMBEDDED_OPTION
        assert "metric_params" in warning[PyEvALLReport.DESCRIPTION_TAG]
