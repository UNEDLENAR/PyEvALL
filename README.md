<p>
    <img src="img/Logo_PyEvALL.png#gh-light-mode-only" height="auto" width="300"/>
    <img src="img/Logo_PyEvALL_dark.png#gh-dark-mode-only" height="auto" width="300"/>
</p>



PyEvALL (The Python to Evaluate ALL) is a evaluation tool for information systems that allows assessing a wide range of metrics covering various evaluation contexts, including classification, ranking, or LeWeDi (Learning with disagreement). PyEvALL is designed based on the following concepts: (i) **persistence**, users can save evaluations and retrieve past evaluations; (ii) **replicability**, all evaluations are conducted using the same methodology, making them strictly comparable; (iii) **effectiveness**, all metrics are unified under measurement theory and have been doubly implemented and compared; (iv) **generalization**, generalization is achieved through the use of a standardized input format enabling users to evaluate all evaluation contexts.

- [What evaluation contexts does PyEvALL include?](#what-evaluation-contexts-does-pyevall-include)
- [Quickstart Guide](#quickstart-guide)
- [What is the input format in PyEvALL?](#what-is-the-input-format-in-pyevall)
   * [Mono-label Classification Format](#mono-label-classification-format)
   * [Multi-label Classification Format](#multi-label-classification-format)
   * [LeWiDi Classification Format](#LeWiDi-classification-format)
   * [Ranking Format](#Ranking-format)

# What evaluation contexts does PyEvALL include?
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
In the example shown, it can be seen that the array consists of three elements belonging to the same test_case, 'EXIST2023', with three different identifiers (I1, I2, and I3), and three different target classes ('A', 'B', and 'C').

## Multi-label Classification Format

The multi-label classification format is one in which each item can be classified with one or several target classes. For this reason, the PyEvALL format for this type is composed of the same elements as in the previous case, with the difference that the "value" attribute in this case is an array of elements. These elements, in turn, must be strings. An example for this format can be found in the following code snippet:

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

The disagreement classification format allows assigning a probability distribution to each class for every element in the dataset. Instead of selecting a single absolute category for each item, the label distribution by annotator is assigned to each element. In this format, as shown in the example below, PyEvALL uses the same structure, except that in this case, the "value" attribute is represented with a dictionary where each element represents a target class and its value represents the probability of assignment. Note that in the case of monolabel disagreement classification, the sum for each element must be 1, while for multilabel classification, it is not necessary.

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

As seen in the example, this shows the predictions of a ranking system indicating that the item with identifier "A" is assigned position 1 in the ranking, while the item with identifier "B" is assigned position 2 in the ranking.
