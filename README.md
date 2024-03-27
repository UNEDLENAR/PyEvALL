<p>
    <img src="img/Logo_PyEvALL.png#gh-light-mode-only" height="auto" width="300"/>
    <img src="img/Logo_PyEvALL_dark.png#gh-dark-mode-only" height="auto" width="300"/>
</p>



PyEvALL (The Python to Evaluate ALL) is a evaluation tool for information systems that allows assessing a wide range of metrics covering various evaluation contexts, including classification, ranking, or LeWeDi (Learning with disagreement). PyEvALL is designed based on the following concepts: (i) **persistence**, users can save evaluations and retrieve past evaluations; (ii) **replicability**, all evaluations are conducted using the same methodology, making them strictly comparable; (iii) **effectiveness**, all metrics are unified under measurement theory and have been doubly implemented and compared; (iv) **generalization**, generalization is achieved through the use of a standardized input format enabling users to evaluate all evaluation contexts.

- [What evaluation contexts does PyEvALL include?](#what-evaluation-contexts-does-pyevall-include)
- [Quickstart Guide](#quickstart-guide)
- [What is the input format in PyEvALL?](#What-is-the-input-format-in-PyEvALL?)

## What evaluation contexts does PyEvALL include?
PyEvALL 2.0 allows evaluation in the following evaluation contexts:

- **Mono-label classification**: evaluation context where each instance is assigned one target class, and only one. Additionally, the classes have no order or hierarchy among them, and all have the same relevance. The available metrics for this context are: *Accuracy, System Precision, Kappa, Precision, Recall, F-Measure, ICM* and *ICM Norm*.

- **Multi-label classification**: evaluation context where each instance is assigned one or more classes from a set of target classes. Additionally, the classes have no order or hierarchy among them, and all have the same relevance. In this context, evaluation can be performed with the metrics *Precision, Recall* and *F-measure*.

- **Mono-label hierarchical classification**: evaluation context where each instance is assigned one target class, and only one. Additionally, the classes have a hierarchical relationship, so that errors between classes at the same hierarchical level represent less failure than errors between classes at different hierarchical levels. In this context, the metrics *ICM* and *ICM Norm* are available.

- **Multi-label hierarchical classification**: Evaluation context where each instance is assigned one or more classes from a set of target classes. Additionally, the classes have a hierarchical relationship, so that errors between classes at the same hierarchical level represent less failure than errors between classes at different hierarchical levels. In this context, the metrics *ICM* and *ICM Norm* can be used.

- **Ranking**: In the ranking evaluation context, the metrics aim to quantify the extent to which a ranking produced by systems is compatible with the relevance values assigned in the gold standard. In this context, the following metrics are available: *Precision at K, R Precision, MRR, MAP, DCG* and *nDCG*.

- **LeWiDi**: Evaluation context where each instance has a probability distribution for all possible classes. To evaluate in disagreement contexts, the metrics *Cross Entropy, ICM Soft* and *ICM Soft Norm* are available.


# Quickstart Guide


# What is the input format in PyEvALL?
The default format for EvALL is JSON due to its great versatility and ease of controlling potential errors according to the format. For this purpose, a schema has been created in which the allowed attributes are defined, as well as their types, declaring for each of them whether they are required or not.

```python
    FORMAT_JSON_SCHEMA= {
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
                            "^.*$": {"type": "number"},    }
                        },
                    ]
                },              
            },
            "required": ["test_case", "id", "value"],
            "additionalProperties": False
        },
    }
```

Specifically, each attribute represents:
- **test_case**: in string format, a specific experiment or use case, which could be, for example, different runs of a classification algorithm or different queries in a ranking context.
- **id**: accepts both string or int format, but it has to be the same for both the predictions and gold standard files. It is the unique identifier of the instance in the dataset.
- **value**: this field represents the value assigned to each item, and its type will vary depending on the applied evaluation context. For example, for mono-label classification, the element will be composed of a string, while for multi-label classification, the element will be composed of a vector of strings.

This format can be adapted for different evaluation contexts. Below are examples of the format based on the context:


