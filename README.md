<p>
    <img src="img/Logo_PyEvALL.png#gh-light-mode-only" height="auto" width="300"/>
    <img src="img/Logo_PyEvALL_dark.png#gh-dark-mode-only" height="auto" width="300"/>
</p>

# PyEvALL

PyEvALL, the Python library to Evaluate ALL, is a Python evaluation tool for information systems. It brings together multiple evaluation contexts in a single framework, including mono-label classification, multi-label classification, hierarchical classification, ranking, LeWiDi (Learning with Disagreement), and clustering.

PyEvALL is designed not only to compute metric scores, but also to make evaluations more reliable. It integrates 30+ metrics, validates input files, detects repeated ids, empty or non-evaluable values, invalid hierarchies, unsupported contexts, and metric precondition failures, and reports these issues explicitly instead of silently producing misleading results.

The library can be used directly from Python code or as part of the EvALL evaluation ecosystem through the EvALL platform: <https://evall.uned.es/>.

Main features:

- Unified evaluation across several task contexts.
- Common JSON input format, with TSV and CSV wrappers converted internally to the same structure.
- Built-in validation of formats, test cases, duplicated ids, empty values, hierarchies, and metric preconditions.
- Simple, embedded, and dataframe reports for programmatic and human inspection.
- Compatibility with the EvALL evaluation platform.

PyEvALL is designed around four principles:

- **Persistence**: evaluations can be stored and inspected later.
- **Replicability**: evaluations follow a common methodology, so results are comparable.
- **Effectiveness**: metrics are implemented under a unified evaluation framework.
- **Generalization**: a standard input schema is used across evaluation contexts.

The canonical PyEvALL input format is JSON. TSV and CSV are also supported as wrappers and are internally converted to the same JSON record structure before validation and evaluation.

## Contents

- [Evaluation Contexts and Metrics](#evaluation-contexts-and-metrics)
- [Quickstart](#quickstart)
- [Input Formats](#input-formats)
- [Evaluation Semantics and Edge Cases](#evaluation-semantics-and-edge-cases)
- [Reports](#reports)
- [Metric Parameters](#metric-parameters)
- [Testing and Validation](#testing-and-validation)

## Evaluation Contexts and Metrics

PyEvALL supports several evaluation contexts. A metric is evaluated only when its input format and evaluation context satisfy the metric preconditions. If a metric is not valid for a context, PyEvALL reports a metric precondition failure instead of returning a misleading value.

### Contexts

- **Mono-label classification**: each instance has one target class.
- **Multi-label classification**: each instance has one or more target classes.
- **Hierarchical classification**: classes are related by a hierarchy. Hierarchical information is used only by metrics that explicitly support it.
- **LeWiDi classification**: each instance has a probability distribution over classes.
- **Ranking**: each prediction assigns a rank/order and the gold standard assigns relevance.
- **Clustering**: each item is assigned to one or more clusters/classes depending on the metric family.

### Metrics

| Metric | Acronym | MetricFactory name | Main context |
|---|---:|---|---|
| Accuracy | Acc | `MetricFactory.Accuracy` | Mono-label classification |
| System Precision | SP | `MetricFactory.SystemPrecision` | Mono-label classification |
| Cohen's Kappa | Kappa | `MetricFactory.Kappa` | Mono-label classification |
| Precision | Pr | `MetricFactory.Precision` | Mono-label and multi-label classification |
| Recall | Re | `MetricFactory.Recall` | Mono-label and multi-label classification |
| F-Measure | F1 | `MetricFactory.FMeasure` | Mono-label and multi-label classification |
| Information Contrast Model | ICM | `MetricFactory.ICM` | Classification, including hierarchical contexts |
| Normalized Information Contrast Model | ICM-Norm | `MetricFactory.ICMNorm` | Classification, including hierarchical contexts |
| Information Contrast Model Soft | ICM-Soft | `MetricFactory.ICMSoft` | LeWiDi, including hierarchical contexts |
| Normalized Information Contrast Model Soft | ICM-Soft-Norm | `MetricFactory.ICMSoftNorm` | LeWiDi, including hierarchical contexts |
| Cross Entropy | CE | `MetricFactory.CrossEntropy` | LeWiDi |
| Mean Absolute Error | MAE | `MetricFactory.MAE` | LeWiDi |
| Precision at k | P@k | `MetricFactory.PrecisionAtK` | Ranking |
| R Precision | RPre | `MetricFactory.RPrecision` | Ranking |
| Mean Reciprocal Rank | MRR | `MetricFactory.MRR` | Ranking |
| Mean Average Precision | MAP | `MetricFactory.MAP` | Ranking |
| Discounted Cumulative Gain | DCG | `MetricFactory.DCG` | Ranking |
| Normalized Discounted Cumulative Gain | nDCG | `MetricFactory.nDCG` | Ranking |
| Expected Reciprocal Rank | ERR | `MetricFactory.ERR` | Ranking |
| Rank Biased Precision | RBP | `MetricFactory.RBP` | Ranking |
| Purity | Pur | `MetricFactory.Purity` | Clustering |
| Inverse Purity | InPur | `MetricFactory.InversePurity` | Clustering |
| F-Measure Purity/Inverse Purity | F(P,I) | `MetricFactory.FMeasurePurityInversePurity` | Clustering |
| Cluster F-Measure | F(P,R) | `MetricFactory.FMeasureClustering` | Clustering |
| Rand Statistics | RandS | `MetricFactory.RandStatistics` | Clustering |
| Jaccard | JACC | `MetricFactory.Jaccard` | Clustering |
| Fowlkes and Mallows | Folk | `MetricFactory.FolkesAndMallows` | Clustering |
| Normalized Mutual Information SG | NMI-SG | `MetricFactory.NMISG` | Clustering |
| Reliability BCubed | R | `MetricFactory.ReliabilityBCubed` | Clustering |
| Sensitivity BCubed | S | `MetricFactory.SensitivityBCubed` | Clustering |
| Entropy | E | `MetricFactory.Entropy` | Clustering |
| Class Entropy | CE | `MetricFactory.ClassEntropy` | Clustering |
| Variation of Information | VI | `MetricFactory.VariationOfInformation` | Clustering |

### Context and metric notes

- `Accuracy`, `SystemPrecision`, and `Kappa` are mono-label classification metrics. They are not evaluated for multi-label classification.
- `Precision`, `Recall`, and `FMeasure` support mono-label and multi-label classification.
- Hierarchical classification is evaluated through hierarchy-aware metrics: `ICM`, `ICMNorm`, `ICMSoft`, and `ICMSoftNorm`. If a hierarchy is provided to a metric that does not use it, the hierarchy is ignored and a warning may be reported.
- `ICM` for multi-label classification without hierarchy is not currently implemented.
- Ranking metrics require numeric integer values. Predictions use ranking/order values; gold uses relevance values.
- LeWiDi metrics require dictionary values where class labels map to numeric probabilities or weights.
- Clustering metrics use list-based values to represent cluster memberships.

## Quickstart

### Installing PyEvALL

PyEvALL can be installed from source or from PyPI:

```bash
pip install PyEvALL
```

### Evaluating a prediction file

The main entry point is `PyEvALLEvaluation.evaluate()`:

```python
def evaluate(self, predictions, goldstandard, lst_metrics, **params):
    """
    predictions: file with system predictions.
    goldstandard: file with ground truth labels.
    lst_metrics: list of metric names.
    params: optional configuration parameters.
    """
```

Example:

```python
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from pyevall.utils.utils import PyEvALLUtils

predictions = "test/resources/metric/test/classification/predictions/SYS5.txt"
gold = "test/resources/metric/test/classification/gold/GOLD5.txt"

metrics = [
    MetricFactory.Accuracy.value,
    MetricFactory.Precision.value,
]

params = {
    PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE,
}

report = PyEvALLEvaluation().evaluate(predictions, gold, metrics, **params)
report.print_report()
```

Metric names should normally be accessed through `MetricFactory`:

```python
metrics = [
    MetricFactory.Accuracy.value,
    MetricFactory.Precision.value,
    MetricFactory.Recall.value,
    MetricFactory.FMeasure.value,
]
```

### Evaluating a list of prediction files

`evaluate_lst()` evaluates several prediction files against the same gold standard:

```python
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from pyevall.utils.utils import PyEvALLUtils

predictions = [
    "test/resources/metric/test/classification/predictions/SYS3.txt",
    "test/resources/metric/test/classification/predictions/SYS4.txt",
    "test/resources/metric/test/classification/predictions/SYS5.txt",
]
gold = "test/resources/metric/test/classification/gold/GOLD5.txt"

metrics = [
    MetricFactory.Accuracy.value,
    MetricFactory.Precision.value,
]

params = {
    PyEvALLUtils.PARAM_REPORT: PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME,
}

report = PyEvALLEvaluation().evaluate_lst(predictions, gold, metrics, **params)
report.print_report()
```

## Input Formats

PyEvALL supports JSON, TSV, and CSV.

- **JSON** is the canonical format.
- **TSV/CSV** are convenience wrappers. They are automatically detected and converted internally to JSON records.
- Gold and prediction files may use different supported formats.
- After TSV/CSV conversion, the same validation rules are applied as for JSON.

### General JSON schema

Every input file is a JSON array of records. Each record contains:

- `test_case`: string identifying the evaluation case, query, task, or run.
- `id`: string identifying the item.
- `value`: the assigned label, score, distribution, rank, relevance, or cluster membership depending on the context.

Allowed `value` types:

- string: mono-label classification.
- array of strings: multi-label classification or clustering membership.
- integer: ranking position or relevance.
- object with numeric values: LeWiDi distributions.

Example schema shape:

```python
FORMAT_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "test_case": {"type": "string"},
            "id": {"type": "string"},
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "integer"},
                    {
                        "type": "object",
                        "patternProperties": {
                            "^.*$": {"type": "number"}
                        }
                    },
                ]
            },
        },
        "required": ["test_case", "id", "value"],
        "additionalProperties": False,
    },
}
```

### TSV and CSV wrappers

TSV and CSV files must contain three columns equivalent to the JSON fields:

```text
test_case	id	value
```

or, without headers:

```text
EXAMPLE	I1	A
EXAMPLE	I2	B
```

PyEvALL detects the format, converts it to JSON records, and reports a warning indicating that a TSV or CSV wrapper was identified.

Important: TSV/CSV in PyEvALL do not define a separate evaluation semantics. Once converted, records follow the same rules as JSON. In particular, repeated `(test_case, id)` pairs are handled by the same repeated-ID validation described below.

### Mono-label classification

Use a string value for each instance:

```json
[
  {
    "test_case": "EXIST2023",
    "id": "I1",
    "value": "A"
  },
  {
    "test_case": "EXIST2023",
    "id": "I2",
    "value": "B"
  }
]
```

### Multi-label classification

Use an array of strings for each instance:

```json
[
  {
    "test_case": "EXIST2023",
    "id": "I1",
    "value": ["A"]
  },
  {
    "test_case": "EXIST2023",
    "id": "I2",
    "value": ["A", "B", "C"]
  }
]
```

If a dataset comes from another tool where multi-label assignments are represented as repeated rows with the same `id`, it must be converted to PyEvALL multi-label JSON by grouping labels in the `value` array. PyEvALL itself treats repeated IDs as a format issue; repeated rows are not interpreted as multi-label assignments.

### LeWiDi classification

Use an object mapping classes to numeric values:

```json
[
  {
    "test_case": "TASK",
    "id": "I1",
    "value": {
      "B": 0.6,
      "C": 0.4
    }
  },
  {
    "test_case": "TASK",
    "id": "I2",
    "value": {
      "B": 0.5,
      "C": 0.5
    }
  }
]
```

### Ranking

Use integer values. In prediction files the integer represents the ranking/order assigned by the system. In gold files the integer represents relevance.

```json
[
  {
    "test_case": "Q1",
    "id": "A",
    "value": 1
  },
  {
    "test_case": "Q1",
    "id": "B",
    "value": 2
  }
]
```

### Clustering

Use an array of strings for cluster memberships:

```json
[
  {
    "test_case": "CLUS",
    "id": "I1",
    "value": ["C1"]
  },
  {
    "test_case": "CLUS",
    "id": "I2",
    "value": ["C1", "C2"]
  }
]
```

Some clustering metrics require non-overlapping cluster assignments, while extended BCubed metrics can work with overlapping assignments. When a metric cannot be evaluated for the provided clustering structure, PyEvALL reports the corresponding precondition failure.

### Hierarchies

Hierarchy information is passed as a parameter, not as part of the input records:

```python
hierarchy = {
    "P": ["S1", "S2"]
}

params[PyEvALLUtils.PARAM_HIERARCHY] = hierarchy
```

For deeper hierarchies, dictionaries can be nested:

```python
hierarchy = {
    "Root": {
        "A": ["A1", "A2"],
        "B": ["B1", "B2"]
    }
}
```

Only hierarchy-aware metrics use this parameter. Other metrics are evaluated without hierarchical semantics.

## Evaluation Semantics and Edge Cases

This section documents the decisions that determine which instances and values are actually evaluated.

### Repeated IDs

PyEvALL expects at most one record for each `(test_case, id)` pair.

- In prediction files, repeated IDs are reported and the repeated row is ignored. Evaluation continues with the first occurrence.
- In gold standard files, repeated IDs are a blocking format error for that file/test case.
- TSV and CSV wrappers follow the same rule because they are converted to the same JSON record structure before validation.

Repeated IDs are not used to represent multi-label assignments in PyEvALL. Multi-label data must use an array in `value`.

### Empty and non-evaluable values

Values such as `""`, `[]`, or `{}` do not represent evaluable assignments.

PyEvALL handles these values as follows:

- A warning is reported: `FORMAT_EMPTY_VALUE_ROW_WARNING`.
- The instance is ignored for evaluation.
- If the empty value appears in gold, the prediction with the same `test_case` and `id` is also removed from the evaluation universe, because there is no valid reference label.
- If the empty value appears in predictions, that prediction is ignored.

Example:

```json
[
  {"test_case": "T1", "id": "I1", "value": ""},
  {"test_case": "T1", "id": "I2", "value": "B"}
]
```

The first row is reported as a warning and excluded from evaluation.

This behavior is intentional: empty values are detected early, reported, and excluded so that metrics are not computed over invalid assignments.

`null` is not a valid `value` according to the PyEvALL JSON schema and is reported as a schema error rather than as an empty-value warning.

### Missing items and missing classes

PyEvALL distinguishes several cases:

- **Missing item**: an item exists in gold but not in predictions, or vice versa.
- **Empty item value**: the item exists but its `value` is empty or non-evaluable.
- **Missing class in gold**: a class appears in predictions but not in gold.
- **Missing class in predictions**: a class appears in gold but not in predictions.

These cases may affect metrics differently. For example, a class-level precision value is reported only for classes in the evaluation context exposed by the comparator. If a class appears only in predictions and not in gold, it may not have a class-level result for some classification metrics.

### Undefined metric values

Some metrics are mathematically undefined in specific situations. PyEvALL reports these cases as `None` in the result or as metric precondition failures, depending on the situation.

Examples:

- Ranking metrics such as `RPrecision`, `MRR`, `MAP`, `nDCG`, `ERR`, and `RBP` may be undefined when the gold standard has no relevant items.
- `Kappa` is not evaluated when the system output has no valid labeled items after empty values are filtered.
- Metrics requiring a context not implemented by PyEvALL return a precondition failure rather than a numeric value.

### Multi-label support

PyEvALL supports multi-label classification for `Precision`, `Recall`, and `FMeasure`.

Metrics such as `Accuracy`, `SystemPrecision`, and `Kappa` are mono-label metrics and are not evaluated for multi-label inputs.

### Hierarchical support

Providing a hierarchy does not make every metric hierarchical. Only hierarchy-aware metrics use it.

Currently, hierarchical evaluation is supported through `ICM` and `ICMNorm` for hard classification labels, and through `ICMSoft` and `ICMSoftNorm` for LeWiDi soft labels. Metrics that do not use hierarchy either ignore it with a warning or fail their context preconditions when the input context is not valid for them.

### Ranking relevance edge cases

When a ranking gold standard has no relevant item, metrics that need at least one relevant item are undefined and return `None`. `PrecisionAtK` may still return a numeric value, because it can be interpreted as the proportion of relevant documents found in the top-k positions.

## Reports

PyEvALL reports are structured around the pair `(prediction file, gold standard file)`.

The report contains:

- `metrics`: metric status, results, errors, warnings, and preconditions.
- `files`: file-level parsing and validation status.

Metric statuses include:

- `OK`: the metric was evaluated.
- `WARNING`: the metric was evaluated but a warning was reported.
- `FAIL`: the metric could not be evaluated.

File statuses follow the same idea. Some file warnings, such as TSV/CSV wrapper detection or empty value filtering, do not necessarily prevent evaluation.

### Embedded report

Use embedded reports to include textual explanations:

```python
params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED
```

### DataFrame report

Use dataframe reports for tabular analysis:

```python
params[PyEvALLUtils.PARAM_REPORT] = PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME
```

DataFrame reports include:

- average results over test cases
- results at test case level
- class-level results when applicable

The embedded report should be inspected whenever a dataframe value is empty, `None`, or `-`.

### Other report outputs

Reports can also be printed as Markdown or TSV:

```python
report = PyEvALLEvaluation().evaluate(predictions, gold, metrics, **params)
report.print_report_to_markdown()
report.print_report_tsv()
```

## Metric Parameters

Some metrics accept parameters through the `params` dictionary.

The structure is:

```python
params = {
    "MetricName": {
        "parameter_name": value
    }
}
```

Example:

```python
params = {
    MetricFactory.FMeasure.value: {"alfa_param": 0.5},
    MetricFactory.ICM.value: {"alpha_1": 2, "alpha_2": 2, "beta": 1},
}
```

Available parameters include:

- **FMeasure**
  - `alfa_param`: weight controlling the balance between precision and recall. Default: `0.5`.
- **ICM**
  - `alpha_1`: contribution of the prediction information content.
  - `alpha_2`: contribution of the gold information content.
  - `beta`: contribution of the union information content.
- **ICMSoft**
  - `alpha_1`
  - `alpha_2`
  - `beta`
- **PrecisionAtK**
  - `k_param`: position up to which precision is calculated.
- **MAP**
  - `r_param`: position up to which average precision is calculated.
- **RBP**
  - `p_param`: persistence parameter.

## Logging

The log level can be configured through parameters:

```python
params[PyEvALLUtils.PARAM_LOG_LEVEL] = PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE
```

Available options:

```python
PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_DEBUG
PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_INFO
PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE
```

## Testing and Validation

PyEvALL includes automated tests for formats and metric comparisons.

### Format tests

Format tests are located in:

```text
test/test_formats.py
```

They cover:

- missing files
- empty files
- invalid JSON
- invalid schema
- duplicated IDs
- TSV/CSV wrapper detection
- type consistency
- empty values and ignored instances
- ranking format errors
- LeWiDi format errors
- clustering format errors
- metric preconditions caused by invalid contexts

Running the format suite generates:

```text
test/reports/test_format_report.html
```

Run:

```bash
python -m pytest test/test_formats.py -q
```

### Metric comparison tests

Metric comparison tests compare PyEvALL results with reference outputs from an external legacy evaluation tool used for validation. That external format is not the PyEvALL input format.

The current comparison suites generate HTML reports:

```text
test/reports/test_classification_metrics_report.html
test/reports/test_ranking_metrics_report.html
test/reports/test_clustering_metrics_report.html
```

Run:

```bash
python -m pytest test/test_classification_metrics.py -q
python -m pytest test/test_ranking_metrics.py -q
python -m pytest test/test_clustering_metrics.py -q
```

Run all tests:

```bash
python -m pytest -q
```

### Test markers

Some format tests can be run by marker:

```bash
python -m pytest -m json_format -q
python -m pytest -m wrappers -q
python -m pytest -m consistency -q
python -m pytest -m clustering -q
```

### Notes on generated HTML reports

The generated HTML reports are intended for development and validation. They help identify:

- matches between PyEvALL and reference outputs
- expected differences
- unsupported contexts
- parser warnings
- precondition failures
- metric values that are intentionally undefined

In restricted environments, writing reports under `test/reports` may require permissions to write to the repository workspace.
