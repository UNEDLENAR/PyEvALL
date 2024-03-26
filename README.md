<p align="center">
    <img src="img/Logo_PyEvALL.png#gh-light-mode-only" height="120" width="120"/>
    <img src="img/Logo_PyEvALL_round.png#gh-dark-mode-only" height="120" width="120"/>
</p>



PyEvALL (The Python to Evaluate ALL) is a evaluation tool for information systems that allows assessing a wide range of metrics covering various evaluation contexts, including classification, ranking, or LeWeDi (Learning with disagreement). PyEvALL is designed based on the following concepts: (i) **persistence**, users can save evaluations and retrieve past evaluations; (ii) **replicability**, all evaluations are conducted using the same methodology, making them strictly comparable; (iii) **effectiveness**, all metrics are unified under measurement theory and have been doubly implemented and compared; (iv) **generalization**, generalization is achieved through the use of a standardized input format enabling users to evaluate all evaluation contexts.

## What evaluation contexts does PyEvALL include?
PyEvALL 2.0 allows evaluation in the following evaluation contexts:

- **Mono-label classification**: evaluation context where each instance is assigned one target class, and only one. Additionally, the classes have no order or hierarchy among them, and all have the same relevance. The available metrics for this context are: *Accuracy, System Precision, Kappa, Precision, Recall, F-Measure, ICM* and *ICM Norm*.

- **Multi-label classification**: evaluation context where each instance is assigned one or more classes from a set of target classes. Additionally, the classes have no order or hierarchy among them, and all have the same relevance. In this context, evaluation can be performed with the metrics *Precision, Recall* and *F-measure*.

- **Mono-label hierarchical classification**: evaluation context where each instance is assigned one target class, and only one. Additionally, the classes have a hierarchical relationship, so that errors between classes at the same hierarchical level represent less failure than errors between classes at different hierarchical levels. In this context, the metrics *ICM* and *ICM Norm* are available.

- **Multi-label hierarchical classification**: Evaluation context where each instance is assigned one or more classes from a set of target classes. Additionally, the classes have a hierarchical relationship, so that errors between classes at the same hierarchical level represent less failure than errors between classes at different hierarchical levels. In this context, the metrics *ICM* and *ICM Norm* can be used.

- **Ranking**: In the ranking evaluation context, the metrics aim to quantify the extent to which a ranking produced by systems is compatible with the relevance values assigned in the gold standard. In this context, the following metrics are available: *Precision at K, R Precision, MRR, MAP, DCG* and *nDCG*.

- **LeWiDi**: Evaluation context where each instance has a probability distribution for all possible classes. To evaluate in disagreement contexts, the metrics *Cross Entropy, ICM Soft* and *ICM Soft Norm* are available.


# Quickstart Guide






