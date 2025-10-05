<p>
    <img src="img/Logo_PyEvALL.png#gh-light-mode-only" height="auto" width="300"/>
    <img src="img/Logo_PyEvALL_dark.png#gh-dark-mode-only" height="auto" width="300"/>
    <img src="img/funding.jpg" height="auto" width="300"/>	
</p>


*Funding:  PyEvALL activities have been partially supported by grantPID2021-124361OB-C32 from the Spanish Ministry of Science and Innovation (funded by MCIN/AEI/10.13039/501100011033 and by ERDF, EU A way of making Europe); and by agreement C039/21-OT-AD2 between UNED and Red.es supported by European Funds (Plan de Recuperación, Transición y Resiliencia).* 

PyEvALL (The Python library to Evaluate ALL) is an evaluation tool for information systems that allows assessing a wide range of metrics covering various evaluation contexts, including classification, ranking, or LeWiDi (Learning with disagreement). PyEvALL is designed based on the following concepts: (i) **persistence**, users can save evaluations and retrieve past evaluations; (ii) **replicability**, all evaluations are conducted using the same methodology, making them strictly comparable; (iii) **effectiveness**, all metrics are unified under measurement theory and have been doubly implemented and compared; (iv) **generalization**, achieved through the use of a standardized input format enabling users to evaluate all evaluation contexts.

<!---
************************		INDEX		************************
-->

- [Implemented Evaluation Contexts and Metrics](#implemented-evaluation-contexts-and-metrics)
- [Quickstart Guide](#quickstart-guide)
   * [Installing PyEvALL](#installing-pyevall)
   * [Evaluating a prediction file](#evaluating-a-prediction-file)
   * [Configuring metric parameters](#Configuring-Parameters-for-the-evaluate-function)  
   * [Input format ](#input-format)
      + [PyEvALL report format parameter](#pyevall-report-format-parameter)
         - [PyEvALL embedded report](#pyevall-embedded-report)
         - [PyEvALL dataframe report](#pyevall-dataframe-report)
      + [Hierarchy parameter](#hierarchy-parameter)
      + [Log Level parameter](#log-level-parameter)
   * [Evaluating a list of prediction files](#evaluating-a-list-of-prediction-files)
- [What is the input format in PyEvALL?](#what-is-the-input-format-in-pyevall)
   * [Mono-label Classification Format](#mono-label-classification-format)
   * [Multi-label Classification Format](#multi-label-classification-format)
   * [LeWiDi Classification Format](#lewidi-classification-format)
   * [Ranking Format](#ranking-format)





<!---
************************		SECTION 1		************************
-->

# Implemented Evaluation Contexts and Metrics
PyEvALL 2.0 allows evaluation in the following contexts:

- **Mono-label classification**: evaluation context where each instance is assigned one target class, and only one. Additionally, the classes have no order or hierarchy among them, and all have the same relevance. The available metrics for this context are: *Accuracy, System Precision, Kappa, Precision, Recall, F-Measure, ICM* and *ICM Norm*.

- **Multi-label classification**: evaluation context where each instance is assigned one or more classes from a set of target classes. Additionally, the classes have no order or hierarchy among them, and all have the same relevance. In this context, evaluation can be performed with the metrics *Precision, Recall* and *F-measure*.

- **Mono-label hierarchical classification**: evaluation context where each instance is assigned one target class, and only one. Additionally, the classes have a hierarchical relationship, so that errors between classes at the same hierarchical level represent less failure than errors between classes at different hierarchical levels. In this context, the metrics *ICM* and *ICM Norm* are available.

- **Multi-label hierarchical classification**: Evaluation context where each instance is assigned one or more classes from a set of target classes. Additionally, the classes have a hierarchical relationship, so that errors between classes at the same hierarchical level represent less failure than errors between classes at different hierarchical levels. In this context, the metrics *ICM* and *ICM Norm* can be used.

- **Ranking**: In the ranking evaluation context, the metrics aim to quantify the extent to which a ranking produced by systems is compatible with the relevance values assigned in the gold standard. In this context, the following metrics are available: *Precision at K, R Precision, MRR, MAP, DCG*, *nDCG*, *ERR* and *RBP*.

- **LeWiDi**: Evaluation context where each instance has a probability distribution for all possible classes. To evaluate in disagreement contexts, the metrics *MAE*, *Cross Entropy, ICM Soft* and *ICM Soft Norm* are available.


|**Metrics**                                   |**Acronym**     |**MetricFactory Name**             |
|------------------------------------------|-------------|-------------------------------|
|Accuracy                                  |Acc          |*MetricFactory.Accuracy*         |
|System Precision                          |SP           |*MetricFactory.SystemPrecision*  |
|Cohen's Kappa                             |Kappa        |*MetricFactory.Kappa*            |
|Precision                                 |Pr           |*MetricFactory.Precision*        |
|Recall                                    |Re           |*MetricFactory.Recall*           |
|F-Measure                                 |F1           |*MetricFactory.FMeasure*         |
|Information Contrast Model                |ICM          |*MetricFactory.ICM*              |
|Normalized Information Contrast Model     |ICM-Norm     |*MetricFactory.ICMNorm*          |
|Information Contrast Model Soft           |ICM-Soft     |*MetricFactory.ICMSoft*          |
|Normalized Information Contrast Model Soft|ICM-Soft-Norm|*MetricFactory.ICMSoftNorm*      |
|Cross Entropy                             |CE           |*MetricFactory.CrossEntropy*     |
|Mean Absolute Error                       |MAE          |*MetricFactory.MAE*     		|
|Precision at k                            |P@k          |*MetricFactory.PrecisionAtK*     |
|R Precision                               |RPre        |*MetricFactory.RPrecision*       |
|Main Reciprocal Rank                      |MRR          |*MetricFactory.MRR*              |
|Mean Average Precision                    |MAP          |*MetricFactory.MAP*              |
|Discounted Cumulative Gain                |DCG          |*MetricFactory.DCG*              |
|Normalized Discounted Cumulative Gain     |nDCG         |*MetricFactory.nDCG*             |
|Expected Reciprocal Rank                  |ERR          |*MetricFactory.ERR*              |
|Rank Biased Precision                     |RBP          |*MetricFactory.RBP*              |



<!---
************************		SECTION 2		************************
-->

# Quickstart Guide
PyEvALL is available via code, by downloading the source or installing the pip package, or via web interface (available soon, July 2024).

## Installing PyEvALL
PyEvALL can be installed via source code, downloading the content of this repository, or installing the [PyEvALL](https://pypi.org/project/PyEvALL/#description)

```python  
pip install PyEvALL

```

## Evaluating a prediction file
The main method to use PyEvALL is the *evaluate()* method included in the class **PyEvALLEvaluation** and whose header is:

```python  
'''
predictions: file in one of the supported formats with the system predictions.
            The default format is JSON.

goldstandard: file in one of the supported formats with the ground truth labels.
              The default format is JSON.

metrics: list with the metrics to evaluate.

params: dictionary with different parameters that can be configured for each evaluation.
'''
def evaluate(self, predictions, goldstandard, lst_metrics, **params):

```
As seen in the code above, the function requires three mandatory parameters and one optional. The mandatory parameters represent the file with the predictions, the file with the gold standard, and the list of metrics to evaluate, respectively. On the other hand, the optional parameter "params" allows configuring different aspects of the evaluation.

An example code of how to evaluate a prediction file is:

```python
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
predictions = "test/resources/metric/test/classification/predictions/SYS5.txt"
gold = "test/resources/metric/test/classification/gold/GOLD5.txt"
test = PyEvALLEvaluation()
metrics=[MetricFactory.Accuracy.value, MetricFactory.Precision.value]

params= dict()
report = test.evaluate(predictions, gold, metrics, **params)
report.print_report()


```

It is important to notice that all metrics' names can be accessed via *MetricFactory* class:

```python
metrics = [MetricFactory.Accuracy.value, MetricFactory.Precision.value, MetricFactory.Recall.value, MetricFactory.FMeasure.value]
```

## Configuring Parameters for the evaluate Function  

The `evaluate` function in the `PyEvALLEvaluation` class enables the evaluation of predictions against a gold standard using various metrics. This function accepts a `params` dictionary that allows specific configurations for each metric. Below, we explain how to structure this dictionary and list the available parameters for each metric.  

### Structure of the `params` Dictionary  

The `params` dictionary should follow this structure:  

```python
params = {
    'MetricName1': {'parameter1': value1, 'parameter2': value2, ...},
    'MetricName2': {'parameter1': value1, ...},
    ...
}
```

For example, to configure the `FMeasure` and `ICM` metrics:  

```python
params = {
    'FMeasure': {'alfa_param': 1.5},
    'ICM': {'alpha_1': 12}
}
```

### Available Parameters for Each Metric  

Here are the configurable parameters for some of the available metrics in PyEvALL:  

- **FMeasure**:  
  - `alfa_param`: Weight that determines the relative importance of precision and recall. 

- **ICM**:  
  - `alpha_1`: Parameter that adjusts the contribution of certain factors in the metric calculation. 
  - `alpha_2`: Parameter that adjusts the contribution of certain factors in the metric calculation. 
  - `beta`: Parameter that adjusts the contribution of certain factors in the metric calculation.   
  
- **ICM Soft**:  
  - `alpha_1`: Parameter that adjusts the contribution of certain factors in the metric calculation. 
  - `alpha_2`: Parameter that adjusts the contribution of certain factors in the metric calculation. 
  - `beta`: Parameter that adjusts the contribution of certain factors in the metric calculation.   

- **Precision at k**:  
  - `k_param`: Position up to which precision is calculated.  

- **MAP**:  
  - `r_param`: Position up to which average precision is calculated.  
  
- **RBP**:  
  - `p_param`: Set the relevance for the metric calculated.   


This section provides a structured reference for configuring the `params` dictionary when using the `evaluate` function, ensuring that metrics are properly customized based on specific needs.


## Input format 
PyEvALL currently supports 3 different formats: JSON, TSV, and CSV, with JSON being the default and primary format. It is important to note that it is not necessary that the files are in the same formats, the gold can be in json and the predictions in tsv. PyEvALL automatically detects the input format and converts it to json to check, by means of its schema, the validation of the input format and informs about the conversion by means of a warning. A detailed desciption of each input format for each evaluation context is done in section [What is the input format in PyEvALL?](#what-is-the-input-format-in-pyevall).



### PyEvALL report format parameter
PyEvALL evaluation reports are generated in JSON format by concatenating different Python dictionaries containing information generated during the evaluation process. This composition of different dictionaries generates a generic JSON used by PyEvALL for its entire internal evaluation process. PyEvALL is designed around the pair *prediction file, gold standard file*, and likewise, the reports focus on this pair.

```python
{
  "metrics": {
    "PrecisionAtK": {
      "name": "Precision at k",
      "acronym": "P@k",
      "description": "Coming soon!",
      "status": "OK",
      "results": {
        "test_cases": [{
          "name": "GOLD1",
          "average": 0.2
        }],
        "average_per_test_case": 0.2
      }
    },
    "Precision": {
      "name": "Precision",
      "acronym": "Pr",
      "description": "Coming soon!",
      "status": "FAIL",
      "results": {
        "test_cases": [],
        "average_per_test_case": null
      },
      "preconditions": {
        "METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION": {
          "name": "METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION",
          "description": "Use parameter: report=\"embedded\"!",
          "status": "FAIL",
          "test_cases": ["GOLD1"]
        }
      }
    }
  },
  "files": {
    "SYS1.txt": {
      "name": "SYS1.txt",
      "status": "OK",
      "gold": false,
      "description": "Use parameter: report=\"embedded\"!",
      "errors": {}
    },
    "GOLD1.txt": {
      "name": "GOLD1.txt",
      "status": "OK",
      "gold": true,
      "description": "Use parameter: report=\"embedded\"!",
      "errors": {}
    }
  }
}
```
In the "metrics" element, the main attributes of the metric are found, which can be useful for generating reports of other types, such as the name or acronym, as well as the results of the metric itself. Additionally, this element includes any potential errors in the analysis of the preconditions of each metric, if any. For example, as seen in the Figure, the metric *precision* has not met its input format precondition for the provided file pair, so it has not been executed, and this is reported to the user.

On the other hand, the "files" element collects any potential errors detected in the files analyzed during the evaluation, i.e., the prediction file and the gold standard file. Each element also includes a description of each error or analysis, allowing for the generation of more explanatory and comprehensive reports. Specifically, to obtain textual explanations of the errors and the analysis of the evaluation process with embedded explanations, the parameter *report=embedded* is used.

The proposed format can be interpreted by various analyzers, providing users with enriched reports tailored to their needs, such as the report with embedded explanations.



#### PyEvALL embedded report
The embedded explanations report of PyEvALL is a JSON report with embedded textual information describing the errors and analysis processes carried out during the evaluation. It provides clear and precise details about the errors detected in the input files. The parameter included should be the following:

```python
params[PyEvALLUtils.PARAM_REPORT]= PyEvALLUtils.PARAM_OPTION_REPORT_EMBEDDED  
```

An example of the embedded explanations report format can be seen in next figure. As shown in the image, this report is almost identical to the previous one except that the description field includes explanations to describe the process. Referring to the previous example, the precision metric generates an error because the input format is not appropriate for this evaluation context, as explained in the message. Likewise, it is indicated that the files have been processed correctly.

```python
{
  "metrics": {
    "PrecisionAtK": {
      "name": "Precision at k",
      "acronym": "P@k",
      "description": "Coming soon!",
      "status": "OK",
      "results": {
        "test_cases": [{
          "name": "GOLD1",
          "average": 0.2
        }],
        "average_per_test_case": 0.2
      }
    },
    "Precision": {
      "name": "Precision",
      "acronym": "Pr",
      "description": "Coming soon!\\nThe evaluation FAIL.",
      "status": "FAIL",
      "results": {
        "test_cases": [],
        "average_per_test_case": null
      },
      "preconditions": {
        "METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION": {
          "name": "METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION",
          "description": " The selected metric cannot be evaluated as the formats of the gold and predictions are not valid for this evaluation context.\\nThe metric name is: Precision.\\nTest case(s) name: GOLD1.",
          "status": "FAIL",
          "test_cases": ["GOLD1"]
        }
      }
    }
  },
  "files": {
    "SYS1.txt": {
      "name": "SYS1.txt",
      "status": "OK",
      "gold": false,
      "description": "The file is correctly parser without errors or warnings.\\nFile name: SYS1.txt.",
      "errors": {}
    },
    "GOLD1.txt": {
      "name": "GOLD1.txt",
      "status": "OK",
      "gold": true,
      "description": "The file is correctly parser without errors or warnings.\\nFile name: GOLD1.txt.",
      "errors": {}
    }
  }
}
```



#### PyEvALL dataframe report
Finally, geared towards a detailed analysis across various metrics, PyEvALL includes the DataFrame report. This report, as its name suggests, consists of multiple Pandas Python library dataframes. Specifically, the report contains 3 dataframes: one with the averages per test case, another with the results disaggregated by test case, and finally another one with the results disaggregated by class, in case any metric operates at the class level. This report can be obtained through code for subsequent analysis, or printed in tabular format for better visualization.

```python
    params[PyEvALLUtils.PARAM_REPORT]= PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME  
```


In this case, the report shown in the image is generated using PyEvALL's method, which enables evaluation given a list of prediction files across a set of metrics. As depicted in the image, these types of reports are often highly useful for in-depth analysis of various models across different metrics, a common practice in the scientific community.


```csv

This is a table PyEvALL report, so no warnings or errors are shown. Please, check the embedded report to check errors if any metric has the value "-" or is an empty value or table.

Table with average results over test cases
+----+----------+----------+------+----------+
|    | files    |      Acc |   Pr |       Re |
|----+----------+----------+------+----------|
|  0 | SYS5.txt | 0.857143 |    1 | 0.666667 |
+----+----------+----------+------+----------+

Table at the test case level
+----+------------+----------+------+----------+
|    | files      |      Acc |   Pr |       Re |
|----+------------+----------+------+----------|
|  0 | SYS5.txt_5 | 0.857143 |    1 | 0.666667 |
+----+------------+----------+------+----------+

Table at the class level
+----+------------+-----------+--------+--------+-----------+--------+--------+
|    | files      |   Pr_TRUE |   Pr_B | Pr_C   |   Re_TRUE |   Re_B |   Re_C |
|----+------------+-----------+--------+--------+-----------+--------+--------|
|  0 | SYS5.txt_5 |         1 |      1 | -      |         1 |      1 |      0 |
+----+------------+-----------+--------+--------+-----------+--------+--------+

```



Moreover, this format enables the generation of reports in other useful formats such as **Markdown** or **tsv**.

```python

report = test.evaluate(predictions, gold, metrics, **params)
report.print_report_to_markdown()

```

```python

report = test.evaluate(predictions, gold, metrics, **params)
report.print_report_tsv()

```

### Hierarchy parameter
PyEvALL allows evaluating with certain metrics that can address hierarchical evaluation contexts, where the set of options available for each item is structured in a hierarchical form. To indicate this hierarchy to the system, this parameter has been introduced, representing the hierarchy through a Python dictionary. To indicate PyEvALL the hierarchical structure of the problem, it is necessary to specify it in the params parameter as follows:

```python

    DIPROMATS_TASK3={"True":{
        "1 appeal to commonality":["1 appeal to commonality - ad populum", "1 appeal to commonality - flag waving"],
        "2 discrediting the opponent":["2 discrediting the opponent - absurdity appeal","2 discrediting the opponent - demonization", "2 discrediting the opponent - doubt", "2 discrediting the opponent - fear appeals (destructive)", "2 discrediting the opponent - name calling", "2 discrediting the opponent - propaganda slinging", "2 discrediting the opponent - scapegoating", "2 discrediting the opponent - undiplomatic assertiveness/whataboutism"],
        "3 loaded language":[], 
        "4 appeal to authority":["4 appeal to authority - appeal to false authority", "4 appeal to authority - bandwagoning"]}, 
        "False":[]}

    params[PyEvALLUtils.PARAM_HIERARCHY]= DIPROMATS_TASK3

```

As shown in the code snippet, the hierarchy is a Python dictionary where each level is formed by a new dictionary, and the leaves are formed by arrays. Note that if the parameter is not specified, metrics that require it cannot be evaluated, or they will be evaluated in a non-hierarchical manner.


### Log Level parameter
The log level can be configured via parameter. PyEvALL defaults to the INFO level and can be changed as shown:

```python

	params [PyEvALLUtils.PARAM_LOG_LEVEL]= PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE

```

The different configuration options for the log level are:

```python

	params [PyEvALLUtils.PARAM_LOG_LEVEL]= PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_DEBUG
	params [PyEvALLUtils.PARAM_LOG_LEVEL]= PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_INFO
	params [PyEvALLUtils.PARAM_LOG_LEVEL]= PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE	

```


## Evaluating a list of prediction files
PyEvALL also provides a method by which a list of prediction files can be evaluated, allowing multiple systems to be evaluated at once. In this mode, PyEvALL generates a meta-report that includes the reports of each *prediction file, gold standard* file pair. The execution of this method would be as follows:

```python
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.utils.utils import PyEvALLUtils
predictions_1 = "test/resources/metric/test/classification/predictions/SYS3.txt"
predictions_2 = "test/resources/metric/test/classification/predictions/SYS4.txt"
predictions_3 = "test/resources/metric/test/classification/predictions/SYS5.txt"
gold = "test/resources/metric/test/classification/gold/GOLD5.txt"
lst_pred= [predictions_1, predictions_2, predictions_3]
test = PyEvALLEvaluation()
metrics=[MetricFactory.Accuracy.value, MetricFactory.Precision.value]

params= dict()
params[PyEvALLUtils.PARAM_FORMAT]= PyEvALLUtils.PARAM_OPTION_FORMAT_TSV
params[PyEvALLUtils.PARAM_REPORT]= PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME 
report = test.evaluate_lst(lst_pred, gold, metrics, **params)
report.print_report()


```

And outputs the following report:

```python
This is a table PyEvALL report, so no warnings or errors are shown. Please, check the embedded report to check errors if any metric has the value "-" or is an empty value or table.

Table with average results over test cases
+----+----------+------------+------+------------+
|    | files    |        Acc |   Pr |         Re |
|----+----------+------------+------+------------|
|  0 | SYS3.txt | nan        |  nan | nan        |
|  1 | SYS4.txt | nan        |  nan | nan        |
|  2 | SYS5.txt |   0.857143 |    1 |   0.666667 |
+----+----------+------------+------+------------+

Table at the test case level
+----+------------+----------+------+----------+
|    | files      |      Acc |   Pr |       Re |
|----+------------+----------+------+----------|
|  0 | SYS5.txt_5 | 0.857143 |    1 | 0.666667 |
+----+------------+----------+------+----------+

Table at the class level
+----+------------+-----------+--------+--------+-----------+--------+--------+
|    | files      |   Pr_TRUE |   Pr_B | Pr_C   |   Re_TRUE |   Re_B |   Re_C |
|----+------------+-----------+--------+--------+-----------+--------+--------|
|  0 | SYS5.txt_5 |         1 |      1 | -      |         1 |      1 |      0 |
+----+------------+-----------+--------+--------+-----------+--------+--------+



```




<!---
************************		SECTION 3		************************
-->


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

This format can be easily obtained form a dataframe by using the option *gold_df.to_json(orient="records")*, like in the following example:

```python
gold_df = pd.DataFrame({'test_case': ['EXIST2023','EXIST2023','EXIST2023','EXIST2023'], 
                        'id': ['101','102','103','104'],
                        'value': ["0","1","0","1"]})
with open(output_file_json, 'w', encoding='utf-8') as output_file: 
            output_file.write(gold_df.to_json(orient="records"))                   
```

This format can be adapted for different evaluation contexts. Below are examples of the format based on the context:

## Mono-label Classification Format

This format is the typical format for mono-label classification tasks where each item has a single associated class. In this format, the label can be any string of characters. An example for this format is:

```python
[
	{  
            "test_case":"EXIST2023",
  	    	 "id":"I1",
            "value":"A"  
          },
          {  
            "test_case":"EXIST2023",
	    	 "id":"I2",
            "value":"B"
          },
          {  
            "test_case":"EXIST2023",
	    	 "id":"I3",
            "value":"C"
          }  
]
```

In the example above, it can be seen that the array consists of three elements belonging to the same test_case, 'EXIST2023', with three different identifiers (I1, I2, and I3), and three different target classes ('A', 'B', and 'C').


## Multi-label Classification Format

In the multi-label classification format, each item can be classified with one or several target classes. For this reason, the PyEvALL format for this type is composed of the same elements as in the previous case, with the difference that the "value" attribute is now an array of elements. These elements, in turn, must be strings. An example for this format can be found in the following code snippet:

```python
[
	  {  
	    "test_case":"EXIST2023",
	    "id":"I1",
	    "value":["A"]  
	  },
	  {  
	    "test_case":"EXIST2023",
	    "id":"I2",
	    "value":["A", "B", "C", "D"] 
	  },
	  {  
	    "test_case":"EXIST2023",
	    "id":"I3",
	    "value":["C", "E"]
	  }
]	
```


As seen in the example, the file consists of an array of JSON objects with three elements with the same fields as in the previous case, but with the difference that, in this case, the "value" attribute is composed of an array with the target classes of each item.

## LeWiDi Classification Format

The disagreement classification format allows assigning a probability distribution to each class for every element in the dataset. Instead of selecting a single absolute category for each item, the label distribution by annotator is assigned to each element. In this format, as shown in the example below, PyEvALL uses the same structure, except that, in this case, the "value" attribute is represented with a dictionary where each element represents a target class and its value represents the probability of assignment. Note that in the case of mono-label disagreement classification, the sum for each element must be 1, while for multi-label classification, it is not necessary.

```python
[
	  {  
	    "test_case":"1",
	    "id":"I1",
	    "value":{
	            "B": 0.6, 
		    "C": 0.4 			
	        } 
	  },
	  {  
	    "test_case":"1",
	    "id":"I2",
	    "value":{
	            "B": 0.5,
	    	    "C": 0.5           
	        } 
	  },  
	  {  
	    "test_case":"1",
	    "id":"I3",
	    "value":{
	            "B": 0.9, 
	            "C": 0.1            
	        } 
	  }
]
```

## Ranking Format

In the ranking evaluation context, each item is assigned a value indicating its position in the ranking for predictions, and the relevance value for the gold standard. As shown in the example below, the format is exactly the same as in the previous cases, but in this case, the value of the "value" attribute consists of numbers representing, in each case, the item's ranking or relevance.

```python
[
	  {  
	    "test_case":"GOLD1",
	    "id":"A",
	    "value":1  
	  },
	  {  
	    "test_case":"GOLD1",
	    "id":"B",
	    "value":2
	  }  
]
```

As seen in the example, this shows the predictions of a ranking system indicating that the item with identifier "A" is assigned the position 1 in the ranking, while the item with identifier "B" is assigned the position 2 in the ranking.
