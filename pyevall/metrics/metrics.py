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
from abc import abstractmethod
from pyevall.metrics.metricfactory  import MetricFactory
from pyevall.reports.reports import PyEvALLReport
from pyevall.utils.utils import PyEvALLUtils
from pyevall.comparators.comparators import PyEvALLComparator
from pyevall.comparators.formats import PyEvALLFormat
import numpy as np
import pandas as pd
from statistics import NormalDist
import math





class PyEvALLMetric(object):    
    def __init__(self, class_name, name, acronym, evaluation_id):
        # Initialize shared metric metadata, results, and report state.
        self.evaluation_id= evaluation_id
        self.logger = PyEvALLUtils.get_logger(__name__, self.evaluation_id)
        self.class_name=class_name
        self.name= name
        self.acronym=acronym
        self.result=dict()
        self.preconditions=dict()
        self.warnings=[]
        self.status = PyEvALLReport.OK   
        
        
    def clear_results(self):
        # Reset computed values and preconditions before reusing the metric.
        self.result=dict()
        self.preconditions=dict()        
    
    
    def add_warning(self, warning):
        if warning is not None and warning not in self.warnings:
            self.warnings.append(warning)

    def log_evaluation_start(self, comparator, message):
        self.logger.info("%s for testcase %s", message, comparator.get_testcase(), stacklevel=2)


    def get_numeric_param(self, params, param_name, default, warning, min_value=None, max_value=None, integer=False):
        if param_name not in params:
            return default
        value = params[param_name]
        valid_type = isinstance(value, int) and not isinstance(value, bool) if integer else isinstance(value, (int, float)) and not isinstance(value, bool)
        if not valid_type:
            self.add_warning(warning)
            return default
        if min_value is not None and value < min_value:
            self.add_warning(warning)
            return default
        if max_value is not None and value > max_value:
            self.add_warning(warning)
            return default
        return value
    
    
    def fire_preconditions(self, error, test_case):
        # Register a failing metric precondition for a test case.
        if error==None:
            return False  
        else:
            if not error in self.preconditions:
                precondition=dict()
                precondition[PyEvALLReport.NAME_TAG]=error
                precondition[PyEvALLReport.DESCRIPTION_TAG]=PyEvALLReport.EMBEDDED_OPTION
                precondition[PyEvALLReport.STATUS_TAG]=PyEvALLReport.FAIL 
                precondition[PyEvALLReport.TEST_CASES_TAG]=[]
                self.preconditions[error]=precondition
            self.preconditions[error][PyEvALLReport.TEST_CASES_TAG].append(test_case)    
            self.status=PyEvALLReport.FAIL      
            
    def fire_warning(self, warning, test_case):
        # Register a non-blocking metric warning for a test case.
        if warning==None:
            return False  
        else:
            if not warning in self.preconditions:
                precondition=dict()
                precondition[PyEvALLReport.NAME_TAG]=warning
                precondition[PyEvALLReport.DESCRIPTION_TAG]=PyEvALLReport.EMBEDDED_OPTION
                precondition[PyEvALLReport.STATUS_TAG]=PyEvALLReport.WARNING 
                precondition[PyEvALLReport.TEST_CASES_TAG]=[]
                self.preconditions[warning]=precondition
            self.preconditions[warning][PyEvALLReport.TEST_CASES_TAG].append(test_case)   
            self.status=PyEvALLReport.WARNING     

         
    @abstractmethod
    def evaluate(self, **params):
        # Evaluate the metric in subclasses.
        raise NotImplementedError("Please Implement this method")
    
    ############################################################################
    ##                                                                        ##
    ##                        CLASSIFICATION METRICS                          ##  
    ##                                                                        ##
    ############################################################################       
class Accuracy(PyEvALLMetric): 
    """
    Accuracy for monolabel classification.

    Formula:
        Accuracy = sum_c TP_c / |G|

    The numerator is the diagonal of the confusion matrix and |G| is the
    number of gold instances.
    """     
    def __init__(self, evaluation_id):        
        # Create the Accuracy metric.
        super().__init__(MetricFactory.Accuracy.value, "Accuracy", "Acc", evaluation_id)    

    def evaluate(self, comparator, **params):  
        # Compute the ratio of correctly classified gold instances.
        self.log_evaluation_start(comparator, "Executing accuracy evaluation method")      
        
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return 
                
        sum_diagonal= comparator.get_diagonal_conf_matrix().sum()
        instances_gold= comparator.get_num_instances_gold()
        acc = sum_diagonal/instances_gold

        self.result[PyEvALLReport.AVERAGE_TAG]=acc
        self.logger.debug("Accuracy for the testcase %s is: %s", comparator.get_testcase(), acc)
            
    def fire_preconditions(self, comparator):
        # Require a monolabel classification comparator.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False

        
                

class SystemPrecision(PyEvALLMetric):  
    """
    System precision for monolabel classification.

    Formula:
        SystemPrecision = sum_c TP_c / |P|

    The numerator is the diagonal of the confusion matrix and |P| is the
    number of predicted instances.
    """     
    def __init__(self, evaluation_id):
        # Create the System Precision metric.
        super().__init__(MetricFactory.SystemPrecision.value, "System Precision", "SP", evaluation_id)        


    def evaluate(self, comparator, **params):   
        # Compute correct predictions over valid predicted instances.
        self.log_evaluation_start(comparator, "Executing System Precision evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
                 
        sp= 0
        sum_diagonal= comparator.get_diagonal_conf_matrix().sum()
        instances_pred= comparator.get_num_instances_pred()
        if not instances_pred==None and not instances_pred==0:
            sp = sum_diagonal/instances_pred
        else:
            sp = None

        self.result[PyEvALLReport.AVERAGE_TAG]=sp
        self.logger.debug("System Precision for the testcase %s is: %s", comparator.get_testcase(), sp)        
           
                
    def fire_preconditions(self, comparator):
        # Require a monolabel classification comparator.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False   
                        
        
                
class Kappa(PyEvALLMetric):         
    """
    Cohen's kappa for monolabel classification.

    Formula:
        Kappa = (Accuracy - Pe) / (1 - Pe)
        Pe = sum_c (Pred_c / |G|) * (Gold_c / |G|)

    Pe is the expected agreement by chance estimated from marginal class
    distributions.
    """

    def __init__(self, evaluation_id):
        # Create the Cohen's Kappa metric.
        super().__init__(MetricFactory.Kappa.value, "Cohen's Kappa", "Kappa", evaluation_id)           
        
        
    def evaluate(self, comparator, **params):
        # Compute chance-corrected agreement from the confusion matrix.
        self.log_evaluation_start(comparator, "Executing kappa evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return

        sum_diagonal= comparator.get_diagonal_conf_matrix().sum()
        instances_gold= comparator.get_num_instances_gold()
        acc = sum_diagonal/instances_gold                              
            
        non_inf_acc=0
        kappa=0    
        classes = comparator.get_classes_gold()            
        for c in classes:
            pred_for_class = comparator.get_pred_for_class(c)      
            instances_gold_class = comparator.get_num_instances_gold_per_category_in_value(c)
            if not instances_gold_class==0:
                non_inf_acc+= (pred_for_class/instances_gold) * (instances_gold_class/instances_gold)                  
            
        if not (1-non_inf_acc)==0:
            kappa = (acc-non_inf_acc)/(1- non_inf_acc)                        
            
        self.result[PyEvALLReport.AVERAGE_TAG]=kappa
            
            
    def fire_preconditions(self, comparator):
        # Require monolabel data with valid predictions and non-degenerate labels.
        error=False
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.get_num_instances_pred()==0:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_EMPTY_PREDICTIONS_ERROR, comparator.get_testcase())
            error=True
        if comparator.is_1_category_in_value_goldAndPred_and_same_instances():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_1_CLASS_GOLDANDPRED_AND_SAME_INSTANCES_ERROR, comparator.get_testcase())
            error=True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return error              
                



class Precision(PyEvALLMetric):        
    """
    Macro precision for monolabel and multilabel classification.

    Formula:
        Precision_c = TP_c / Pred_c
        Precision = average_c Precision_c

    Classes with no predicted instances receive an undefined per-class value
    and are skipped in the macro average.
    """

    def __init__(self, evaluation_id):
        # Create the macro Precision metric.
        super().__init__(MetricFactory.Precision.value, "Precision", "Pr", evaluation_id)           
        
        
    def evaluate(self, comparator, **params):
        # Compute per-class precision and average over defined classes.
        self.log_evaluation_start(comparator, "Executing precision evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return

        classes = comparator.get_classes_gold()
        self.result[PyEvALLReport.CLASSES_TAG]=dict()
        num_classes=0
        aveg_class=0
        for c in classes:
            TP = comparator.get_true_positive_per_class(c)
            instances_pred_class= comparator.get_num_instances_pred_per_category_in_value(c)

            if (not instances_pred_class==None) and (not instances_pred_class==0):
                p= TP/instances_pred_class
                self.result[PyEvALLReport.CLASSES_TAG][c]=p
                aveg_class += p
                num_classes+= 1
                self.logger.debug("Precision for class "+ c + " in testcase " + comparator.get_testcase() + " is: " + str(p))
            else:
                self.result[PyEvALLReport.CLASSES_TAG][c]=None
                self.logger.debug("Precision for class "+ c + " in testcase " + comparator.get_testcase() + " not exist ")                        
                
        aveg=None
        if not num_classes==0:
            aveg = aveg_class/num_classes
    
        self.result[PyEvALLReport.AVERAGE_TAG]=aveg


    def fire_preconditions(self, comparator):
        # Require monolabel or multilabel classification data.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL] or
                     comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL])):  
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False     
                        
                
            
                    
class Recall(PyEvALLMetric):          
    """
    Macro recall for monolabel and multilabel classification.

    Formula:
        Recall_c = TP_c / Gold_c
        Recall = average_c Recall_c
    """

    def __init__(self, evaluation_id):
        # Create the macro Recall metric.
        super().__init__(MetricFactory.Recall.value, "Recall", "Re", evaluation_id)          
        
        
    def evaluate(self, comparator, **params):
        # Compute per-class recall and macro-average it.
        self.log_evaluation_start(comparator, "Executing recall evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return

        classes = comparator.get_classes_gold()
        self.result[PyEvALLReport.CLASSES_TAG]=dict()
        aveg_class = 0
        for c in classes:
            TP = comparator.get_true_positive_per_class(c)
            instances_gold_class= comparator.get_num_instances_gold_per_category_in_value(c)
            r= TP/instances_gold_class
            self.result[PyEvALLReport.CLASSES_TAG][c]=r
            aveg_class += r
            self.logger.debug("Recall for class "+ c + " in testcase " + comparator.get_testcase() + " is: " + str(r))                      
                
        aveg = aveg_class/len(classes)
        self.result[PyEvALLReport.AVERAGE_TAG]=aveg


    def fire_preconditions(self, comparator):
        # Require monolabel or multilabel classification data.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL] or
                     comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL])):   
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())              
        return False       
               
                
                
        
class FMeasure(PyEvALLMetric):     
    """
    Macro F-measure for monolabel and multilabel classification.

    Formula:
        F_c = 1 / (alpha / Precision_c + (1 - alpha) / Recall_c)
        F = average_c F_c

    With alpha=0.5 this is the usual F1 harmonic mean.
    """

    def __init__(self, evaluation_id):
        # Create the macro F-measure metric.
        super().__init__(MetricFactory.FMeasure.value, "F-Measure", "F1", evaluation_id)  
        self.alfa_param=0.5          
                
                
    def evaluate(self, comparator, **params):
        # Compute weighted harmonic F-measure per class and average it.
        self.log_evaluation_start(comparator, "Executing fmeasure evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return 

        self.alfa_param = self.get_numeric_param(params, "alfa_param", self.alfa_param, PyEvALLReport.METRIC_PARAM_ALFA_NOT_VALID_WARNING, 0, 1)
        classes = comparator.get_classes_gold()
        self.result[PyEvALLReport.CLASSES_TAG]=dict()
        aveg_class = 0
        for c in classes:
            TP = comparator.get_true_positive_per_class(c)
            instances_gold_class= comparator.get_num_instances_gold_per_category_in_value(c)
            instances_pred_class= comparator.get_num_instances_pred_per_category_in_value(c)                
            if (not instances_pred_class==None) and (not instances_pred_class==0):
                p= TP/instances_pred_class
                r= TP/instances_gold_class
                if not p==0 and not r==0:
                    f1 = 1/((self.alfa_param/p) + ((1-self.alfa_param)/r));                        
                    self.result[PyEvALLReport.CLASSES_TAG][c]=f1
                    aveg_class += f1
                    self.logger.debug("F1 for class "+ str(c) + " in testcase " + comparator.get_testcase() + " is: " + str(f1))
                else:
                    self.result[PyEvALLReport.CLASSES_TAG][c]=0
            else:
                self.result[PyEvALLReport.CLASSES_TAG][c]=0             
                
        aveg = aveg_class/len(classes)
        self.result[PyEvALLReport.AVERAGE_TAG]=aveg   
    
        
    def fire_preconditions(self, comparator):
        # Require monolabel or multilabel classification data.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL] or
                     comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL])):  
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False

        


class ICM(PyEvALLMetric):   
    """
    Information Contrast Model for hard classification labels.

    Formula:
        ICM_i = alpha_1 * IC(P_i) + alpha_2 * IC(G_i) - beta * IC(P_i union G_i)
        ICM = average_i ICM_i

    IC(S) is computed recursively from class probabilities, subtracting the
    information content of deepest common ancestors for hierarchical labels.
    """

    def __init__(self, evaluation_id):
        # Create the ICM metric and initialize its probability caches.
        super().__init__(MetricFactory.ICM.value, "Information Contrast model", "ICM", evaluation_id)
        #parameters icm
        self.alpha_1=2
        self.alpha_2=2
        self.beta=3  
        
        #data structures for probabilities
        self.gold_freq= dict() 
        self.gold_prob= dict()      
        
        
    def generate_prob(self, comparator):
        # Estimate class probabilities from gold labels and optional hierarchy.
        #Mono label classification
        if comparator.proporties[comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]:
            #Flat classification
            if comparator.hierarchy==None:
                self.gold_freq= comparator.gold_df[PyEvALLFormat.VALUE].value_counts().to_dict() 
                gold_size = len(comparator.gold_df)
                for c in self.gold_freq:
                    self.gold_prob[c]= self.gold_freq[c]/gold_size                    
                    
            #Hierarchical classification
            else:
                self.gold_freq= comparator.gold_df[PyEvALLFormat.VALUE].value_counts().to_dict() 
                gold_size = len(comparator.gold_df)
            
                self.calculate_prob_hierarchy_mono_label(comparator.hierarchy)            
                for c in self.gold_freq:
                    self.gold_prob[c]= self.gold_freq[c]/gold_size                    
        
        #Multi label classification            
        elif comparator.proporties[comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]:
            if comparator.hierarchy==None:
                raise NotImplementedError("Please Implement this method")
            else:  
                #check for classses not included in hierarchy   
                comparator.gold_df.apply(lambda row: self.check_class_not_in_hierachy(row, comparator.hierarchy), axis=1)           
                gold_size = len(comparator.gold_df)
                
                self.calculate_prob_hierarchy_multi_labe(comparator.hierarchy, comparator)           
                for c in self.gold_freq:
                    self.gold_prob[c]= self.gold_freq[c]/gold_size  
    
    
    ######################################################################
    #                                                                    #
    #    Methods to calculate probabilities to compute ICM  metric.      #
    #                                                                    #
    ######################################################################       
    def calculate_prob_hierarchy_mono_label(self, hierarchy):
        # Propagate monolabel class frequencies upward through the hierarchy.
        if len(hierarchy)==0:
            return 0
        
        level_freq=0
        if isinstance(hierarchy, dict):
            for c in hierarchy:
                freq_c = 0 if c not in self.gold_freq else self.gold_freq[c]
                value = freq_c + self.calculate_prob_hierarchy_mono_label(hierarchy[c])
                self.gold_freq[c]=value
                level_freq+=value
        elif isinstance(hierarchy, list):
            for c in hierarchy:
                freq_c = 0 if c not in self.gold_freq else self.gold_freq[c]                
                self.gold_freq[c]=freq_c
                level_freq+=freq_c
            
        return level_freq
    
    
    def check_class_not_in_hierachy(self, gold_row, hierarchy):
        # Add gold labels missing from the hierarchy as leaf nodes.
        gold_set=[]
        if np.isscalar(gold_row[PyEvALLFormat.VALUE]):
            gold_set.append(gold_row[PyEvALLFormat.VALUE])
        else:
            gold_set=gold_row[PyEvALLFormat.VALUE]  
            
        for c in gold_set:
            if not self.is_child(hierarchy, c):
                hierarchy[c]=[]     
    
    
    def calculate_prob_hierarchy_multi_labe(self, hierarchy,comparator):
        # Estimate multilabel hierarchical frequencies for every hierarchy node.
        if len(hierarchy)==0:
            return 0
        
        if isinstance(hierarchy, dict):
            for c in hierarchy:
                freq_c = sum(comparator.gold_df.apply(lambda row: self.belgons_item_to_class_or_subclass(row, c, hierarchy[c]), axis=1).tolist())
                self.gold_freq[c]=freq_c
                self.calculate_prob_hierarchy_multi_labe(hierarchy[c], comparator)
                
        elif isinstance(hierarchy, list):
            for c in hierarchy:
                freq_c = sum(comparator.gold_df.apply(lambda row: self.belgons_item_to_class_or_subclass(row, c, []), axis=1).tolist())
                self.gold_freq[c]=freq_c        
        
    
    def belgons_item_to_class_or_subclass(self, gold_row, clas, hierarchy):
        # Return whether a gold item belongs to a class or any descendant.
        gold_set=[]
        if np.isscalar(gold_row[PyEvALLFormat.VALUE]):
            gold_set.append(gold_row[PyEvALLFormat.VALUE])
        else:
            gold_set=gold_row[PyEvALLFormat.VALUE]
            
        if clas in gold_set:
            return 1
        else:
            for c in gold_set:
                if self.is_child(hierarchy, c):
                    return 1
        return 0        
        
        
    def is_child(self, hierarchy, child):
        # Check recursively whether a class appears in a hierarchy subtree.
        if isinstance(hierarchy, dict):
            exist = False
            for c in hierarchy:
                if c ==child:
                    return True
                else:
                    exist= exist or self.is_child(hierarchy[c], child)
            return exist    
        elif isinstance(hierarchy, list):
            return child in hierarchy   
   
  
    ######################################################################
    #                                                                    #
    #    Methods to process the metric ICM                               #
    #                                                                    #
    ###################################################################### 
    def evaluate(self, comparator, **params):   
        # Compute average ICM over all gold instances.
        self.log_evaluation_start(comparator, "Executing ICM evaluation method")
        if self.fire_preconditions(comparator):
            return
        

        self.alpha_1 = self.get_numeric_param(params, "alpha_1", self.alpha_1, PyEvALLReport.METRIC_PARAM_ALPHA_1_NOT_VALID_WARNING)
        self.alpha_2 = self.get_numeric_param(params, "alpha_2", self.alpha_2, PyEvALLReport.METRIC_PARAM_ALPHA_2_NOT_VALID_WARNING)
        self.beta = self.get_numeric_param(params, "beta", self.beta, PyEvALLReport.METRIC_PARAM_BETA_NOT_VALID_WARNING)
         
        self.generate_prob(comparator) 
        
        result_icm = comparator.gold_df.apply(lambda row: self.calculate_icm_row(row,comparator), axis=1).tolist()

        gold_size = len(comparator.gold_df)
        average = sum(result_icm)/gold_size

        self.result[PyEvALLReport.AVERAGE_TAG]=average

        
    def fire_preconditions(self, comparator):
        # Require supported hard-label classification context.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL] or
                     comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL])):  
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] and 
            comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]
            and comparator.hierarchy==None):
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_IMPLEMENTED_EVALUATION_CONTEXT, comparator.get_testcase())
            return True            
        return False 
    
    
    def calculate_icm_row(self, gold_row, comparator):
        # Compute the ICM contribution for one gold instance.
        pred_row= comparator.pred_df[comparator.pred_df[PyEvALLFormat.ID] == gold_row[PyEvALLFormat.ID]]
        pred_set=[]
        gold_set=[]        

        if not pred_row.empty:
            if np.isscalar(pred_row[PyEvALLFormat.VALUE].tolist()[0]):
                pred_set.append(pred_row[PyEvALLFormat.VALUE].tolist()[0])
            else:
                pred_set=pred_row[PyEvALLFormat.VALUE].tolist()[0]
            
        if np.isscalar(gold_row[PyEvALLFormat.VALUE]):
            gold_set.append(gold_row[PyEvALLFormat.VALUE])
        else:
            gold_set=gold_row[PyEvALLFormat.VALUE]
        
        union_set= list(set(pred_set) | set(gold_set))         
        return self.alpha_1*self.information_content(pred_set, comparator) + self.alpha_2*self.information_content(gold_set, comparator) - self.beta*self.information_content(union_set, comparator)

                
    def information_content(self, classes, comparator):
        # Compute recursive information content for a set of hard labels.
        size = len(classes)
        if size==0:
            return 0
        return self.get_prob_class(classes[0],comparator) + self.information_content(classes[1:size],comparator) - \
        self.information_content(self.calculate_set_deepest_common_ancestor( classes[0], classes[1:size], comparator), comparator)
                    
    
    def get_prob_class(self, clas,comparator):
        # Return the information value of a class from the gold distribution.
        #Empty set
        if clas==None:
            return 0
                
        #Class does not exist in gold we add minimal information
        if (not clas in self.gold_prob) or (self.gold_prob[clas]==0.0):
            self.gold_prob[clas]=1/len(comparator.gold_df)
            return -math.log2(self.gold_prob[clas])        
        
        else:
            return -math.log2(self.gold_prob[clas])
        
     
    def calculate_set_deepest_common_ancestor(self, clas, classes, comparator):
        # Find deepest common ancestors between a class and a class set.
        deepest_common_ancestors=[]   
            
        for c in classes:
            parents_a= self.get_parents_dict(comparator.hierarchy, clas)
            parents_b= self.get_parents_dict(comparator.hierarchy, c)
            if parents_a==None or parents_b==None:
                continue
            #intersection list parents, only common are save
            common= [ e for e in parents_a if e in parents_b ]
            size= len(common)
            #select only the deepest parent
            common = common[size-1:]
            #Union with previous deepest parents
            deepest_common_ancestors= list(set(deepest_common_ancestors) | set(common))

        return deepest_common_ancestors      
        
        
    def get_parents_dict(self, nested_dict, value):
        # Return the path from hierarchy root to a class value.
        if nested_dict == value:
            return [nested_dict]
        elif isinstance(nested_dict, dict):
            for k, v in nested_dict.items():
                if k == value:
                    return [k]
                p = self.get_parents_dict(v, value)
                if p:
                    return [k] + p
        elif isinstance(nested_dict, list):
            lst = nested_dict
            for i in range(len(lst)):
                p = self.get_parents_dict(lst[i], value)
                if p:
                    return p        
                
                
                
                
class ICMNorm(PyEvALLMetric):       
    """
    Normalized Information Contrast Model.

    Formula:
        ICMNorm = (ICM(P, G) + ICM(G, G)) / (2 * ICM(G, G))

    This is an approximate normalization around the perfect self-evaluation
    score ICM(G, G). Since ICM is not naturally bounded and can range from
    negative infinity to positive infinity, a strict min-max normalization is
    not possible without defining a theoretical worst case. Values below 0 are
    truncated to 0, but values above 1 may appear when the system score falls
    outside the interval implicitly assumed by this approximation.
    """

    def __init__(self, evaluation_id):
        # Create the normalized ICM metric.
        super().__init__(MetricFactory.ICMNorm.value, "Normalized Information Contrast Model", "ICM-Norm", evaluation_id)
       
    
    def evaluate(self, comparator, **params):   
        # Normalize ICM by comparing predictions against the gold self-score.
        self.log_evaluation_start(comparator, "Executing ICM Normalized evaluation method")
        if self.fire_preconditions(comparator):
            return
        
        #Evaluate ICM pred vs gold
        icm = ICM(self.evaluation_id)
        icm.evaluate(comparator, **params)
        res_pred= 0
        if PyEvALLReport.AVERAGE_TAG in icm.result:
            res_pred = icm.result[PyEvALLReport.AVERAGE_TAG] 
                   
        #Evaluate ICM gold vs gold  
        comp_gold = PyEvALLComparator(comparator.gold_df, comparator.gold_df, comparator.get_testcase(), self.evaluation_id)
        comp_gold.hierarchy= comparator.hierarchy
        icm = ICM(self.evaluation_id)
        icm.evaluate(comp_gold, **params)
        res_gold=0
        if PyEvALLReport.AVERAGE_TAG in icm.result:
            res_gold=icm.result[PyEvALLReport.AVERAGE_TAG]
        
        #Calculate ICM Norm and truncate to 0 if the value is less than 0.
        icm_norm= (float(res_pred) - (res_gold*-1))/(res_gold-(res_gold*-1))
        if icm_norm<0:
            icm_norm=0

        self.result[PyEvALLReport.AVERAGE_TAG]=icm_norm

        
    def fire_preconditions(self, comparator):
        # Require supported hard-label classification context.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL] or
                     comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL])):  
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] and 
            comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]
            and comparator.hierarchy==None):
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_IMPLEMENTED_EVALUATION_CONTEXT, comparator.get_testcase())
            return True            
        return False 




    ############################################################################
    ##                                                                        ##
    ##                        CLASSIFICATION LEWEDI METRICS                   ##  
    ##                                                                        ##
    ############################################################################
class ICMSoft(PyEvALLMetric): 
    """
    Information Contrast Model for LeWiDi soft labels.

    Formula:
        ICMSoft_i = alpha_1 * IC(P_i) + alpha_2 * IC(G_i) - beta * IC(P_i union G_i)
        ICMSoft = average_i ICMSoft_i

    Soft information content is estimated from each class distribution in the
    gold data, using the normal CDF over class weights.
    """

    def __init__(self, evaluation_id):
        # Create the soft-label ICM metric and initialize distribution caches.
        super().__init__(MetricFactory.ICMSoft.value, "Information Contrast Model Soft", "ICM-Soft", evaluation_id)     
        #parameters icm
        self.alpha_1=2
        self.alpha_2=2
        self.beta=3 
               
        #data structures for probabilities
        self.gold_average= dict()
        self.gold_deviation= dict()
        self.lst_classes= []  

  
    ######################################################################
    #                                                                    #
    #    Methods to calculate probabilities to compute ICM  soft metric. #
    #                                                                    #
    ######################################################################      
    def get_list_classes(self, comparator):
        # Collect all classes from gold distributions or from the hierarchy.
        if comparator.hierarchy==None:
            comparator.gold_df[PyEvALLFormat.VALUE].apply(lambda value: self.search_classes(value))
        else:
            #check for classses not included in hierarchy   
            comparator.gold_df.apply(lambda row: self.check_class_not_in_hierachy(row, comparator.hierarchy), axis=1)            
            self.get_classes_hierarchy(comparator.hierarchy)
            
    
    def search_classes(self, value):
        # Add classes from one LeWiDi distribution to the metric cache.
        gold_dict = value
        for c in gold_dict:
            if c not in self.lst_classes:
                self.lst_classes.append(c)  
                  
    
    def check_class_not_in_hierachy(self, gold_row, hierarchy):
        # Add soft-label classes missing from the hierarchy as leaves.
        gold_set=[]
        if np.isscalar(gold_row[PyEvALLFormat.VALUE]):
            gold_set.append(gold_row[PyEvALLFormat.VALUE])
        else:
            gold_set=gold_row[PyEvALLFormat.VALUE]  
            
        for c in gold_set:
            if not self.is_child(hierarchy, c):
                hierarchy[c]=[]      
       
                        
    def is_child(self, hierarchy, child):
        # Check recursively whether a class appears in a hierarchy subtree.
        if isinstance(hierarchy, dict):
            exist = False
            for c in hierarchy:
                if c ==child:
                    return True
                else:
                    exist= exist or self.is_child(hierarchy[c], child)
            return exist    
        elif isinstance(hierarchy, list):
            return child in hierarchy 
            
            
    def get_classes_hierarchy(self, hierarchy):
        # Flatten all classes in a hierarchy into the metric class list.
        if isinstance(hierarchy, dict):            
            for c in hierarchy: 
                self.lst_classes.append(c)             
                self.get_classes_hierarchy(hierarchy[c])  
        elif isinstance(hierarchy, list):
            for c in hierarchy:      
                self.lst_classes.append(c) 
                      
    
    def calculate_probabilities(self, comparator):  
        # Estimate mean and deviation for each soft-label class.
        gold_df_extended= comparator.gold_df.copy()
        
        if not comparator.hierarchy==None:          
            gold_df_extended[PyEvALLFormat.VALUE] = gold_df_extended[PyEvALLFormat.VALUE].apply(lambda row: self.propagate_max_weigth_ancestors(row.copy(), comparator.hierarchy, None))
        
        gold_df_extended[self.lst_classes] =gold_df_extended[PyEvALLFormat.VALUE].apply(lambda row: self.expand_df(row))
        gold_df_extended = gold_df_extended.drop(PyEvALLFormat.VALUE, axis=1)
            
        sum_items = gold_df_extended.sum().tolist()
        size_gold = len(gold_df_extended)       
        for ind, column in enumerate(gold_df_extended.columns):
            if column==PyEvALLFormat.ID or column==PyEvALLFormat.TEST_CASE:
                continue
            
            self.gold_average[column] = sum_items[ind]/size_gold
            dict_deviation= gold_df_extended[column].value_counts().to_dict()
            deviation= 0
            for v in dict_deviation:
                items_dev = dict_deviation[v] 
                val = float(v)
                deviation = deviation + abs(val-self.gold_average[column])* items_dev
                                
            self.gold_deviation[column]= deviation/size_gold                   
   
 
    def propagate_max_weigth_ancestors(self, gold_dict, hierarchy, clas):
        # Propagate soft-label weights from descendants to ancestors.
        if isinstance(hierarchy, dict):            
            for c in hierarchy:                
                self.propagate_max_weigth_ancestors(gold_dict, hierarchy[c], c)
                if not clas==None: 
                    if clas not in gold_dict:
                        gold_dict[clas]= gold_dict[c]
                    else:
                        gold_dict[clas]= max(gold_dict[clas], gold_dict[c])
        elif isinstance(hierarchy, list):
            for c in hierarchy: 
                if c not in gold_dict:
                    gold_dict[c]=0.0                           
                if clas not in gold_dict:
                    gold_dict[clas]= gold_dict[c]
                else:
                    gold_dict[clas]= max(gold_dict[clas], gold_dict[c]) 
                
        return gold_dict  
               
            
    def expand_df(self, value):
        # Expand one soft-label distribution into a fixed class vector.
        gold_dict = value
        new_columns=[]
        for c in self.lst_classes:
            if c in gold_dict:
                new_columns.append(gold_dict[c])
            else:
                new_columns.append(0.0)

        return pd.Series(new_columns, index=[self.lst_classes])         
        
 
     
    ######################################################################
    #                                                                    #
    #            Methods to process the metric ICM soft                  #
    #                                                                    #
    ######################################################################    
    def evaluate(self, comparator, **params):   
        # Compute average soft ICM over all gold instances.
        self.log_evaluation_start(comparator, "Executing ICM Soft evaluation method")
        if self.fire_preconditions(comparator):
            return
          
        self.alpha_1 = self.get_numeric_param(params, "alpha_1", self.alpha_1, PyEvALLReport.METRIC_PARAM_ALPHA_1_NOT_VALID_WARNING)
        self.alpha_2 = self.get_numeric_param(params, "alpha_2", self.alpha_2, PyEvALLReport.METRIC_PARAM_ALPHA_2_NOT_VALID_WARNING)
        self.beta = self.get_numeric_param(params, "beta", self.beta, PyEvALLReport.METRIC_PARAM_BETA_NOT_VALID_WARNING)               
        self.get_list_classes(comparator)
        self.calculate_probabilities(comparator)
        
        result_icm_soft = comparator.gold_df.apply(lambda row: self.calculate_icm_row(row, comparator), axis=1).tolist()
        gold_size = len(comparator.gold_df)
        average = sum(result_icm_soft)/gold_size
        
        self.result[PyEvALLReport.AVERAGE_TAG]=average    

        
    def fire_preconditions(self, comparator):
        # Require a LeWiDi classification comparator.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        return False     
    
        
    def calculate_icm_row(self, gold_row, comparator):
        # Compute soft ICM contribution for one gold instance.
        pred_row= comparator.pred_df[comparator.pred_df[PyEvALLFormat.ID] == gold_row[PyEvALLFormat.ID]]
        pred_set=[]
        gold_set=[]
        
        if not pred_row.empty:
            pred_dict = pred_row[PyEvALLFormat.VALUE].tolist()[0]
            for c in pred_dict:
                pred_set.append((c,pred_dict[c]))
        
        gold_dict = gold_row[PyEvALLFormat.VALUE]
        for c in gold_dict:
            gold_set.append((c,gold_dict[c]))           
        
        union_set= self.union_soft(gold_set, pred_set)
        return self.alpha_1*self.information_content(pred_set, comparator) + self.alpha_2*self.information_content(gold_set, comparator) - self.beta*self.information_content(union_set, comparator)
    
    
    def union_soft(self, set_a, set_b):
        # Build the soft union of two weighted class sets using max weights.
        union = []
        for a in set_a:
            union.append(a)
            
        for b in set_b:
            exist=False
            for i, u in enumerate(union):
                if b[0]==u[0]:
                    val =max(b[1], u[1])
                    clas = u[0]
                    union[i]= (clas, val)
                    exist = True
                
            if not exist:
                union.append(b)
        
        return union
                
    
    def information_content(self, classes, comparator):
        # Compute recursive information content for weighted class tuples.
        size = len(classes)
        if size==0:
            return 0
        return self.get_prob_class(classes[0], comparator) + self.information_content(classes[1:size], comparator) - \
        self.information_content(self.calculate_set_deepest_common_ancestor(classes[0], classes[1:size], comparator), comparator)
                    
    
    def get_prob_class(self, tupla,comparator):
        # Return the information value of a weighted class tuple.
        #Empty set
        if tupla==None or not tupla[0]:
            return 0
        
        #Class does not exist in gold we add minimal information
        if not tupla[0] in self.gold_average:
            return -math.log2(1/len(comparator.gold_df))        
        else:
            if tupla[1]==0.0:
                return 0.0
            else:
                prob = 1-NormalDist(mu=self.gold_average[tupla[0]], sigma=self.gold_deviation[tupla[0]]).cdf(tupla[1])  
                if prob==0.0:
                    return -math.log2(1/len(comparator.gold_df)) 
                else:         
                    return -math.log2(prob)    
        
    
    def calculate_set_deepest_common_ancestor(self, clas, classes, comparator):
        # Find deepest common ancestors for weighted soft-label classes.
        deepest_common_ancestors=[]           
        for c in classes:
            parents_a= self.get_parents_dict(comparator.hierarchy, clas[0])
            parents_b= self.get_parents_dict(comparator.hierarchy, c[0])
            if parents_a==None or parents_b==None:
                continue
            #intersection list parents, only common are save
            common= [ e for e in parents_a if e in parents_b ]
            size= len(common)
            #select only the deepest parent 
            if size!=0:
                tupla=(common[size-1:][0], min(clas[1], c[1]))
                common = [tupla]
                #Union with previous deepest parents
                deepest_common_ancestors= self.union_soft(deepest_common_ancestors, common) 
                
        return deepest_common_ancestors


    def get_parents_dict(self, nested_dict, value):
        # Return the path from hierarchy root to a class value.
        if nested_dict == value:
            return [nested_dict]
        elif isinstance(nested_dict, dict):
            for k, v in nested_dict.items():
                if k == value:
                    return [k]
                p = self.get_parents_dict(v, value)
                if p:
                    return [k] + p
        elif isinstance(nested_dict, list):
            lst = nested_dict
            for i in range(len(lst)):
                p = self.get_parents_dict(lst[i], value)
                if p:
                    return p   




class ICMSoftNorm(PyEvALLMetric):       
    """
    Normalized Information Contrast Model for LeWiDi soft labels.

    Formula:
        ICMSoftNorm = (ICMSoft(P, G) + ICMSoft(G, G)) / (2 * ICMSoft(G, G))

    This is an approximate normalization around the perfect self-evaluation
    score ICMSoft(G, G). Since ICMSoft is not naturally bounded and can range
    from negative infinity to positive infinity, a strict min-max normalization
    is not possible without defining a theoretical worst case. Values below 0
    are truncated to 0, but values above 1 may appear when the system score
    falls outside the interval implicitly assumed by this approximation.
    """

    def __init__(self, evaluation_id):
        # Create the normalized soft ICM metric.
        super().__init__(MetricFactory.ICMSoftNorm.value, "Normalized Information Contrast Model Soft", "ICM-Soft-Norm", evaluation_id)      

   
    def evaluate(self, comparator, **params):   
        # Normalize soft ICM by comparing predictions against the gold self-score.
        self.log_evaluation_start(comparator, "Executing ICM-Soft Normalized evaluation method")
        if self.fire_preconditions(comparator):
            return
        
        #Evaluate ICM pred vs gold
        icm_soft = ICMSoft(self.evaluation_id)
        icm_soft.evaluate(comparator, **params)
        res_pred= 0
        if PyEvALLReport.AVERAGE_TAG in icm_soft.result:
            res_pred = icm_soft.result[PyEvALLReport.AVERAGE_TAG]        
            
        #Evaluate ICM gold vs gold       
        comp_gold = PyEvALLComparator(comparator.gold_df, comparator.gold_df, comparator.get_testcase(), self.evaluation_id)
        comp_gold.hierarchy= comparator.hierarchy
        icm_soft = ICMSoft(self.evaluation_id)
        icm_soft.evaluate(comp_gold, **params)
        res_gold=0
        if PyEvALLReport.AVERAGE_TAG in icm_soft.result:
            res_gold=icm_soft.result[PyEvALLReport.AVERAGE_TAG]
        
        #Calculate ICM Norm and truncate to 0 if the value is less than 0.
        icm_norm= (float(res_pred) - (res_gold*-1))/(res_gold-(res_gold*-1))
        if icm_norm<0:
            icm_norm=0

        self.result[PyEvALLReport.AVERAGE_TAG]=icm_norm

        
    def fire_preconditions(self, comparator):
        # Require a LeWiDi classification comparator.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        return False 





class CrossEntropy(PyEvALLMetric): 
    """
    Cross entropy for LeWiDi soft labels.

    Formula:
        CE = average_i -sum_c G_i(c) * log2(P_i(c))

    Gold and prediction distributions are smoothed and normalized before the
    instance-level cross entropy is computed.
    """

    SMOOTH_VALUE=0.001
    
    
    def __init__(self, evaluation_id):
        # Create the Cross Entropy metric.
        super().__init__(MetricFactory.CrossEntropy.value, "Cross Entropy", "CE", evaluation_id)
       
       
    def smooth_and_normalize_data(self, gold_df, pred_df):           
        # Smooth and normalize gold and prediction distributions in place.
        gold_df[PyEvALLFormat.VALUE] = gold_df[PyEvALLFormat.VALUE].apply(lambda row: self.normalize_data(self.smooth_data(row.copy())))   
        pred_df[PyEvALLFormat.VALUE] = pred_df[PyEvALLFormat.VALUE].apply(lambda row: self.normalize_data(self.smooth_data(row.copy())))
        
        
    def smooth_data(self, row):  
        # Replace non-positive probabilities with the smoothing constant.
        set=[]
        if np.isscalar(row):
            set.append(row)
        else:
            set=row
            
        for c in set:
            if set[c]<=0.0:
                set[c]=CrossEntropy.SMOOTH_VALUE  
                
        return set
                
    
    def normalize_data(self, row):  
        # Normalize a probability dictionary so its values sum to one.
        set=[]
        if np.isscalar(row):
            set.append(row)
        else:
            set=row  
            
        sum= 0.0            
        for c in set:
            sum = sum+ set[c]
        
        for c in set:
            set[c]=set[c]/sum
            
        return set                          
      
    
    def evaluate(self, comparator, **params):
        # Compute average cross entropy over all gold instances.
        self.log_evaluation_start(comparator, "Executing Cross Entropy evaluation method")
        if self.fire_preconditions(comparator):
            return
        pred_df = comparator.pred_df.copy()
        gold_df = comparator.gold_df.copy() 
        self.smooth_and_normalize_data(gold_df, pred_df)           
        
        result_cross_entropy = gold_df.apply(lambda row: self.calculate_cross_entropy_instance(row, pred_df), axis=1).tolist()
        gold_size = len(gold_df)
        average = sum(result_cross_entropy)/gold_size
        
        self.result[PyEvALLReport.AVERAGE_TAG]=average    
       
      
    def fire_preconditions(self, comparator):
        # Require LeWiDi data and warn when hierarchy is ignored.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())  
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False 


    def calculate_cross_entropy_instance(self, gold_row, pred_df):
        # Compute cross entropy for one gold instance.
        pred_row= pred_df[pred_df[PyEvALLFormat.ID] == gold_row[PyEvALLFormat.ID]]
        
        pred_dict=[]
        if not pred_row.empty:
            pred_dict = pred_row[PyEvALLFormat.VALUE].tolist()[0]
        
        gold_dict = gold_row[PyEvALLFormat.VALUE]
       
        cross_entropy_instance=0.0
        for c in gold_dict:
            if c in pred_dict:
                cross_entropy_instance= cross_entropy_instance+ (gold_dict[c]* math.log2(pred_dict[c]))               
        
        
        return cross_entropy_instance*-1


class MAE(PyEvALLMetric):    
    """
    Mean Absolute Error for LeWiDi soft labels.

    Formula:
        MAE_i = (1 / |C|) * sum_c |G_i(c) - P_i(c)|
        MAE = average_i MAE_i
    """

    def __init__(self, evaluation_id):
        # Create the Mean Absolute Error metric.
        super().__init__(MetricFactory.MAE.value, "Mean Absolute Error", "MAE", evaluation_id)  
        self.lst_classes=[] 


    def get_list_classes(self, comparator):
        # Collect all classes appearing in gold LeWiDi distributions.
        comparator.gold_df[PyEvALLFormat.VALUE].apply(lambda value: self.search_classes(value))

    def search_classes(self, value):
        # Add classes from one distribution to the metric class list.
        gold_dict = value
        for c in gold_dict:
            if c not in self.lst_classes:
                self.lst_classes.append(c)  

    def expand_df(self, value):
        # Expand one distribution into a vector over all gold classes.
        gold_dict = value
        new_columns=[]
        for c in self.lst_classes:
            if c in gold_dict:
                new_columns.append(gold_dict[c])
            else:
                new_columns.append(0.0)

        return pd.Series(new_columns, index=[self.lst_classes])  

    def evaluate(self, comparator, **params):   
        # Compute average MAE over expanded LeWiDi distributions.
        self.log_evaluation_start(comparator, "Executing MAE evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
        
        self.get_list_classes(comparator)        
        
        #copy dataframe modify it
        gold_df_extended= comparator.gold_df.copy()  
        gold_df_extended[self.lst_classes] =gold_df_extended[PyEvALLFormat.VALUE].apply(lambda row: self.expand_df(row))
        gold_df_extended = gold_df_extended.drop(PyEvALLFormat.VALUE, axis=1)         
        
        number_classes= len(self.lst_classes)
        result_mae = gold_df_extended.apply(lambda row: self.calculate_mae_instance(row, comparator.pred_df, number_classes), axis=1).tolist()
        gold_size = len(gold_df_extended)
        average = sum(result_mae)/gold_size
        
        self.result[PyEvALLReport.AVERAGE_TAG]=average  
             
                
    def fire_preconditions(self, comparator):
        # Require LeWiDi data and warn when hierarchy is ignored.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())  
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False 


    def calculate_mae_instance(self, gold_row, pred_df, number_classes):    
        # Compute MAE for one expanded gold instance.
        pred_row= pred_df[pred_df[PyEvALLFormat.ID] == gold_row[PyEvALLFormat.ID]]
        mae_instance=0.0        
           
        for ind, column in enumerate(gold_row.index):
            if column==PyEvALLFormat.ID or column==PyEvALLFormat.TEST_CASE:
                continue
            
            score_pred= 0
            if not pred_row.empty:
                pred_dict=pred_row[PyEvALLFormat.VALUE].tolist()[0]
                if column in pred_dict: 
                    score_pred= pred_row[PyEvALLFormat.VALUE].tolist()[0][column]            
            mae_instance= mae_instance+ abs(gold_row[column]-score_pred)
            
            
            
        if ind>0:
            mae_instance=mae_instance/number_classes
            
        return mae_instance



    ############################################################################
    ##                                                                        ##
    ##                        RANKING METRICS                                 ##  
    ##                                                                        ##
    ############################################################################     
class PrecisionAtK(PyEvALLMetric):     
    """
    Precision at rank k.

    Formula:
        P@k = relevant_items_in_top_k / k
    """

    def __init__(self, evaluation_id):
        # Create Precision at K with its default cutoff.
        super().__init__(MetricFactory.PrecisionAtK.value, "Precision at k", "P@k", evaluation_id)    
        self.k_param=10     


    def evaluate(self, comparator, **params):   
        # Compute relevant items retrieved in the first k predicted ranks.
        self.log_evaluation_start(comparator, "Executing Precision at K evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
              
        self.k_param = self.get_numeric_param(params, "k_param", self.k_param, PyEvALLReport.METRIC_PARAM_K_NOT_VALID_WARNING, 0, integer=True)                 
        p_at_k=None
        relevants = comparator.get_first_k_relevant_items_in_pred(self.k_param)
        if self.k_param!=0:
            p_at_k = relevants/self.k_param

        self.result[PyEvALLReport.AVERAGE_TAG]=p_at_k
        self.logger.debug("Precision at k for the testcase %s is: %s", comparator.get_testcase(), p_at_k)        
                
                
    def fire_preconditions(self, comparator):
        # Require a ranking comparator and warn about tied prediction values.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False     
    
    
    
    
class RPrecision(PyEvALLMetric):       
    """
    R-Precision.

    Formula:
        RPrecision = relevant_items_in_top_R / R

    R is the number of relevant items in the gold ranking.
    """

    def __init__(self, evaluation_id):
        # Create the R-Precision metric.
        super().__init__(MetricFactory.RPrecision.value, "R Precision", "RPre", evaluation_id)    


    def evaluate(self, comparator, **params):   
        # Compute precision at R, where R is the number of relevant gold items.
        self.log_evaluation_start(comparator, "Executing R Precision evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
                 
        r_p=None
        param_k = comparator.get_all_relevants_items_in_gold()
        relevants = comparator.get_first_k_relevant_items_in_pred(param_k)
        if param_k!=0:
            r_p = relevants/param_k

        self.result[PyEvALLReport.AVERAGE_TAG]=r_p
        self.logger.debug("R Precision for the testcase %s is: %s", comparator.get_testcase(), r_p)        
                
                
    def fire_preconditions(self, comparator):
        # Require a ranking comparator and warn about tied prediction values.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False  
    
    
    
    
class MRR(PyEvALLMetric):    
    """
    Mean Reciprocal Rank for one evaluated ranking.

    Formula:
        MRR = 1 / rank_first_relevant

    If the gold data contains no relevant items, the score is undefined.
    """

    def __init__(self, evaluation_id):
        # Create the MRR metric.
        super().__init__(MetricFactory.MRR.value, "Mean Reciprocal Rank", "MRR", evaluation_id)    

    def evaluate(self, comparator, **params):   
        # Compute reciprocal rank of the first relevant predicted item.
        self.log_evaluation_start(comparator, "Executing Main Reciprocal Rank evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
                 
        #check if there are relevants elements in the gold
        if comparator.get_all_relevants_items_in_gold()>0:
            mmr=0
            position_relevant = comparator.get_position_first_relevant_in_pred()
            if position_relevant!=None:
                mmr = 1/position_relevant
        else:
            mmr=None

        self.result[PyEvALLReport.AVERAGE_TAG]=mmr
        self.logger.debug("MRR for the testcase %s is: %s", comparator.get_testcase(), mmr)        
                
                
    def fire_preconditions(self, comparator):
        # Require a ranking comparator and warn about tied prediction values.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]):  
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False      
    
    
    
    
class MAP(PyEvALLMetric):       
    """
    Mean Average Precision for one evaluated ranking.

    Formula:
        MAP = (1 / R) * sum_k P@k * rel(k)

    R is the number of relevant gold items and rel(k) is 1 when the item at
    rank k is relevant.
    """

    def __init__(self, evaluation_id):
        # Create MAP with its default ranking cutoff.
        super().__init__(MetricFactory.MAP.value, "Mean Average Precision", "MAP", evaluation_id) 
        self.r_param=1000   


    def evaluate(self, comparator, **params):   
        # Compute average precision over relevant predicted positions.
        self.log_evaluation_start(comparator, "Executing MAP evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
        
        self.r_param = self.get_numeric_param(params, "r_param", self.r_param, PyEvALLReport.METRIC_PARAM_R_NOT_VALID_WARNING, 0, integer=True)         
        relevants_in_gold= comparator.get_all_relevants_items_in_gold()   
        lst_ids_pred_sorted = comparator.get_list_ids_ordered_pred() 
        threshold = min(self.r_param,len(lst_ids_pred_sorted))
        sumPrecision=0
        
        for k in range(0,threshold):
            relevants_in_k= comparator.get_first_k_relevant_items_in_pred(k+1)
            if comparator.is_relevant_item_with_id_in_gold(lst_ids_pred_sorted[k]):
                sumPrecision+=relevants_in_k/(k+1)
             
        map=None
        if relevants_in_gold!=None and relevants_in_gold!=0:
            map = sumPrecision/relevants_in_gold            

        self.result[PyEvALLReport.AVERAGE_TAG]=map
        self.logger.debug("MAP for the testcase %s is: %s", comparator.get_testcase(), map)        
              
                
    def fire_preconditions(self, comparator):
        # Require a ranking comparator and warn about tied prediction values.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase()) 
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False     
    
    
    
    
class DCG(PyEvALLMetric):     
    """
    Discounted Cumulative Gain.

    Formula:
        DCG = sum_k (2^rel_k - 1) / log2(k + 1)

    Ranks are one-based in the formula.
    """

    def __init__(self, evaluation_id):
        # Create the DCG metric.
        super().__init__(MetricFactory.DCG.value, "Discounted Cumulative Gain", "DCG", evaluation_id)   


    def evaluate(self, comparator, **params):   
        # Compute discounted gain over the predicted ranking.
        self.log_evaluation_start(comparator, "Executing DCG evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
        
        values_gold= comparator.get_dict_values_gold_by_id()  
        lst_ids_pred_sorted = comparator.get_list_ids_ordered_pred() 
        threshold = len(lst_ids_pred_sorted)        
        
        #check if there are relevants elements in the gold
        if comparator.get_all_relevants_items_in_gold()>0:
            dcg=0
            for k in range(0,threshold):
                id_pred = lst_ids_pred_sorted[k]
                if id_pred in values_gold:
                    log_item= math.log10((k+1)+1)/math.log10(2)
                    if not (log_item==0):
                        numerator= pow(2, values_gold[id_pred])-1
                        dcg= dcg + (numerator/log_item)

        else:
            dcg=None
            
        self.result[PyEvALLReport.AVERAGE_TAG]=dcg
        self.logger.debug("DCG for the testcase %s is: %s", comparator.get_testcase(), dcg)        
              
                
    def fire_preconditions(self, comparator):
        # Require a ranking comparator and warn about tied prediction values.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase()) 
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False 
    
    
    
    
class nDCG(PyEvALLMetric):    
    """
    Normalized Discounted Cumulative Gain.

    Formula:
        nDCG = DCG / IDCG

    IDCG is the DCG obtained from the ideal ranking sorted by gold relevance.
    """

    def __init__(self, evaluation_id):
        # Create the nDCG metric.
        super().__init__(MetricFactory.nDCG.value, "Normalized Discounted Cumulative Gain", "nDCG", evaluation_id)   


    def evaluate(self, comparator, **params):   
        # Compute DCG normalized by the ideal gold ranking DCG.
        self.log_evaluation_start(comparator, "Executing nDCG evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
        
        # Compute DCG over the predicted ranking.
        values_gold= comparator.get_dict_values_gold_by_id()  
        lst_ids_pred_sorted = comparator.get_list_ids_ordered_pred() 
        threshold = len(lst_ids_pred_sorted)
        sum_dcg=0
        
        for k in range(0,threshold):
            id_pred = lst_ids_pred_sorted[k]
            if id_pred in values_gold:
                log_item= math.log10((k+1)+1)/math.log10(2)
                if not (log_item==0):
                    numerator= pow(2, values_gold[id_pred])-1
                    sum_dcg= sum_dcg + (numerator/log_item)
        
        
        #Calculate dcg over the perfect ranking
        perfect_ranking = comparator.get_list_perfect_rank_gold() 
        threshold = len(perfect_ranking)
        sum_idcg=0
        
        for k in range(0,threshold):
            id_pred = perfect_ranking[k]
            if id_pred in values_gold:
                log_item= math.log10((k+1)+1)/math.log10(2)
                if not (log_item==0):
                    numerator= pow(2, values_gold[id_pred])-1
                    sum_idcg= sum_idcg + (numerator/log_item)        
             
        ndcg=None
        if not (sum_idcg==0):
            ndcg=sum_dcg/sum_idcg            

        self.result[PyEvALLReport.AVERAGE_TAG]=ndcg
        self.logger.debug("nDCG for the testcase %s is: %s", comparator.get_testcase(), ndcg)   
             
                
    def fire_preconditions(self, comparator):
        # Require a ranking comparator and warn about tied prediction values.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase()) 
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False    
        
    
    
class ERR(PyEvALLMetric):    
    """
    Expected Reciprocal Rank.

    Formula:
        ERR = sum_k (REL_k / k) * product_j<k (1 - REL_j)
        REL_k = (2^rel_k - 1) / 2^max_rel

    Ranks are one-based in the formula.
    """

    def __init__(self, evaluation_id):
        # Create the ERR metric.
        super().__init__(MetricFactory.ERR.value, "Expected Reciprocal Rank", "ERR", evaluation_id)   


    def evaluate(self, comparator, **params):   
        # Compute expected reciprocal rank with graded relevance.
        self.log_evaluation_start(comparator, "Executing ERR evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
        
        #if there are no relevants in gold it makes no sense calculate the metric
        relevants_in_gold= comparator.get_all_relevants_items_in_gold()  
        if relevants_in_gold==None or relevants_in_gold==0:
            self.result[PyEvALLReport.AVERAGE_TAG]=None
            return 
       
        # get the max value in the gold     
        values_gold= comparator.get_dict_values_gold_by_id()
        perfect_rank= comparator.get_list_perfect_rank_gold()
        maxValue = 0 if len(perfect_rank)==0 else values_gold[perfect_rank[0]] 
        powMaxValue =0
        if maxValue>0:
            powMaxValue= pow(2, maxValue)
        lst_ids_pred_sorted = comparator.get_list_ids_ordered_pred()
         
        
        err=0
        for k in range(0,len(lst_ids_pred_sorted)):
            id_pred = lst_ids_pred_sorted[k]
            RELk = 0
            errMulti=1
            if id_pred in values_gold:
                for j in range(0,k):
                    RELi=0
                    id_pred_j = lst_ids_pred_sorted[j]
                    if id_pred_j in values_gold:
                        if powMaxValue!=0.0:
                            RELi = (pow(2, values_gold[id_pred_j])-1)/powMaxValue
                    errMulti= errMulti*(1-RELi);
                if powMaxValue!=0.0:
                    RELk = (pow(2, values_gold[id_pred])-1)/powMaxValue;
                    
            err=err +(RELk/(float)(k+1))* errMulti;                     


        self.result[PyEvALLReport.AVERAGE_TAG]=err
        self.logger.debug("ERR for the testcase %s is: %s", comparator.get_testcase(), err)   
             
                
    def fire_preconditions(self, comparator):
        # Require a ranking comparator and warn about tied prediction values.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase()) 
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False        
    
    
    
class RBP(PyEvALLMetric):    
    """
    Rank Biased Precision.

    Formula:
        RBP = (1 - p) * sum_k rel_k * p^(k - 1)

    The persistence parameter p defaults to 0.8.
    """

    def __init__(self, evaluation_id):
        # Create RBP with its default persistence parameter.
        super().__init__(MetricFactory.RBP.value, "Rank Biased Precision", "RBP", evaluation_id)   
        self.p_param=0.8 


    def evaluate(self, comparator, **params):   
        # Compute rank-biased precision over the predicted ranking.
        self.log_evaluation_start(comparator, "Executing RBP evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
        
        self.p_param = self.get_numeric_param(params, "p_param", self.p_param, PyEvALLReport.METRIC_PARAM_P_NOT_VALID_WARNING, 0, 1)  
        #if there are no relevants in gold it makes no sense calculate the metric
        relevants_in_gold= comparator.get_all_relevants_items_in_gold()  
        if relevants_in_gold==None or relevants_in_gold==0:
            self.result[PyEvALLReport.AVERAGE_TAG]=None
            return 
       
        # get the max value in the gold     
        values_gold= comparator.get_dict_values_gold_by_id()
        lst_ids_pred_sorted = comparator.get_list_ids_ordered_pred()       
        
        rbpAux=0
        for k in range(0,len(lst_ids_pred_sorted)):
            id_pred = lst_ids_pred_sorted[k]            
            posI= k+1;
            if id_pred in values_gold:
                rbpAux = rbpAux + values_gold[id_pred]* pow(self.p_param, posI-1)
                    
        rbp = (1-self.p_param)*rbpAux;   
        self.result[PyEvALLReport.AVERAGE_TAG]=rbp
        self.logger.debug("RBP for the testcase %s is: %s", comparator.get_testcase(), rbp)   
             
                
    def fire_preconditions(self, comparator):
        # Require a ranking comparator and warn about tied prediction values.
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase()) 
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())             
        return False 






############################################################################
##                                                                        ##
##                        CLUSTERING METRICS                              ##  
##                                                                        ##
############################################################################
class Purity(PyEvALLMetric):
    """
    Cluster purity.

    Formula:
        Purity(C, L) = sum_c (|c| / N) * max_l |c intersect l| / |c|

    Only predicted items assigned to a valid gold class are used in the
    cluster-side counts, matching the legacy clustering evaluation behavior.
    """

    def __init__(self, evaluation_id):
        # Create the Purity clustering metric.
        super().__init__(MetricFactory.Purity.value, "Purity", "Pur", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute weighted cluster purity over valid predicted clusters.
        self.log_evaluation_start(comparator, "Executing purity evaluation method")
        if self.fire_preconditions(comparator):
            return

        purity = 0.0
        clusters_pred = comparator.get_all_clusters_pred()
        clusters_gold = comparator.get_all_clusters_gold()
        # Use the gold-labeled universe so extra prediction-only items do not affect purity.
        valid_gold_ids = comparator.get_valid_gold_cluster_items()
        total_instances_with_repetitions = comparator.get_valid_pred_cluster_occurrences(clusters_pred, valid_gold_ids)
        if total_instances_with_repetitions != 0:
            for cluster_pred in clusters_pred:
                # Weight each predicted cluster by its valid size and its best class precision.
                size_cluster = len(comparator.get_valid_pred_cluster_items(cluster_pred, valid_gold_ids))
                if size_cluster > 0:
                    purity += (size_cluster / total_instances_with_repetitions) * comparator.get_max_filtered_precision_for_pred_cluster(cluster_pred, clusters_gold, valid_gold_ids)

        self.result[PyEvALLReport.AVERAGE_TAG] = purity

    def fire_preconditions(self, comparator):
        # Require a clustering comparator.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        return False


class InversePurity(PyEvALLMetric):
    """
    Inverse purity.

    Formula:
        InversePurity(C, L) = sum_l (|l| / N) * max_c |c intersect l| / |l|

    It rewards assigning the items of each gold class to a common predicted
    cluster.
    """

    def __init__(self, evaluation_id):
        # Create the Inverse Purity clustering metric.
        super().__init__(MetricFactory.InversePurity.value, "Inverse Purity", "InPur", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute weighted inverse purity over gold classes.
        self.log_evaluation_start(comparator, "Executing inverse purity evaluation method")
        if self.fire_preconditions(comparator):
            return

        inverse_purity = 0.0
        total_instances_with_repetitions = comparator.get_num_cluster_occurrences_gold()
        clusters_pred = comparator.get_all_clusters_pred()
        if total_instances_with_repetitions != 0:
            for cluster_gold in comparator.get_all_clusters_gold():
                # Weight each gold class by its size and its best cluster recall.
                size_cluster = comparator.get_num_items_for_cluster_in_gold(cluster_gold)
                if size_cluster > 0:
                    inverse_purity += (size_cluster / total_instances_with_repetitions) * comparator.get_max_precision_for_gold_cluster(cluster_gold, clusters_pred)

        self.result[PyEvALLReport.AVERAGE_TAG] = inverse_purity

    def fire_preconditions(self, comparator):
        # Require a clustering comparator.
        # Inverse purity is defined only for clustering-format comparators.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        return False


class FMeasurePurityInversePurity(PyEvALLMetric):
    """
    Harmonic combination of Purity and Inverse Purity.

    Formula:
        F(P, IP) = 1 / (alpha / P + (1 - alpha) / IP)

    The balance is computed after obtaining the global purity and inverse
    purity values.
    """

    def __init__(self, evaluation_id):
        # Create the Purity/Inverse Purity F-measure metric.
        super().__init__(MetricFactory.FMeasurePurityInversePurity.value, "F Measure", "F(P, I)", evaluation_id)
        self.alfa_param = 0.5

    def evaluate(self, comparator, **params):
        # Compute the harmonic combination of purity and inverse purity.
        self.log_evaluation_start(comparator, "Executing fmeasure purity/inverse purity evaluation method")
        if self.fire_preconditions(comparator):
            return

        self.alfa_param = self.get_numeric_param(params, "alfa_param", self.alfa_param, PyEvALLReport.METRIC_PARAM_ALFA_NOT_VALID_WARNING, 0, 1)

        purity = 0.0
        inverse_purity = 0.0
        total_gold_with_repetitions = comparator.get_num_cluster_occurrences_gold()
        clusters_pred = comparator.get_all_clusters_pred()
        clusters_gold = comparator.get_all_clusters_gold()
        # Reuse the same valid prediction universe as Purity.
        valid_gold_ids = comparator.get_valid_gold_cluster_items()
        total_pred_with_repetitions = comparator.get_valid_pred_cluster_occurrences(clusters_pred, valid_gold_ids)

        # Compute the purity component.
        if total_pred_with_repetitions != 0:
            for cluster_pred in clusters_pred:
                size_cluster = len(comparator.get_valid_pred_cluster_items(cluster_pred, valid_gold_ids))
                if size_cluster > 0:
                    purity += (size_cluster / total_pred_with_repetitions) * comparator.get_max_filtered_precision_for_pred_cluster(cluster_pred, clusters_gold, valid_gold_ids)

        # Compute the inverse purity component.
        if total_gold_with_repetitions != 0:
            for cluster_gold in clusters_gold:
                size_cluster = comparator.get_num_items_for_cluster_in_gold(cluster_gold)
                if size_cluster > 0:
                    inverse_purity += (size_cluster / total_gold_with_repetitions) * comparator.get_max_precision_for_gold_cluster(cluster_gold, clusters_pred)

        f_measure_pi = 0.0
        # Harmonic combination; undefined zero components produce a zero score.
        if purity != 0.0 and inverse_purity != 0.0:
            denominator = (self.alfa_param / purity) + ((1 - self.alfa_param) / inverse_purity)
            if denominator != 0.0:
                f_measure_pi = 1 / denominator

        self.result[PyEvALLReport.AVERAGE_TAG] = f_measure_pi

    def fire_preconditions(self, comparator):
        # Require a clustering comparator.
        # The metric is defined only for clustering-format comparators.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        return False


class FMeasureClustering(PyEvALLMetric):
    """
    Cluster F-measure.

    Formula:
        F(C, L) = sum_l (|l| / N) * max_c F(c, l)
        F(c, l) = 2 * Precision(c, l) * Recall(c, l) / (Precision(c, l) + Recall(c, l))

    Unlike the Purity/Inverse Purity combination, precision and recall are
    balanced for each cluster-class pair before selecting the best match.
    """

    def __init__(self, evaluation_id):
        # Create the cluster F-measure metric.
        super().__init__(MetricFactory.FMeasureClustering.value, "F Measure", "F(P, R)", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute best matching cluster F-measure for each gold class.
        self.log_evaluation_start(comparator, "Executing fmeasure clustering evaluation method")
        if self.fire_preconditions(comparator):
            return

        fmeasure = 0.0
        total_instances_with_repetitions = comparator.get_num_cluster_occurrences_gold()
        clusters_pred = comparator.get_all_clusters_pred()
        # Precision uses only predicted items with a valid gold class.
        valid_gold_ids = comparator.get_valid_gold_cluster_items()
        if total_instances_with_repetitions != 0:
            for cluster_gold in comparator.get_all_clusters_gold():
                # Weight each gold class by the best pairwise F score found in predictions.
                size_cluster = comparator.get_num_items_for_cluster_in_gold(cluster_gold)
                if size_cluster > 0:
                    fmeasure += (size_cluster / total_instances_with_repetitions) * comparator.get_max_filtered_f_for_gold_cluster(cluster_gold, clusters_pred, valid_gold_ids)

        self.result[PyEvALLReport.AVERAGE_TAG] = fmeasure

    def fire_preconditions(self, comparator):
        # Require a clustering comparator.
        # The metric is defined only for clustering-format comparators.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        return False


class RandStatistics(PyEvALLMetric):
    """
    Rand index for clustering pair decisions.

    Formula:
        Rand = (SS + DD) / (SS + SD + DS + DD)

    SS/DD are pairs where gold and prediction agree; SD/DS are disagreements.
    """

    def __init__(self, evaluation_id):
        # Create the Rand Statistics metric.
        super().__init__(MetricFactory.RandStatistics.value, "Rand Statistics", "RandS", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute Rand index from clustering pair agreements.
        self.log_evaluation_start(comparator, "Executing rand statistics evaluation method")
        if self.fire_preconditions(comparator):
            return

        # Compute the pair confusion table over valid gold items.
        pairs = comparator.get_pair_confusion_clustering()
        numerator = pairs["SS"] + pairs["DD"]
        denominator = pairs["SS"] + pairs["SD"] + pairs["DS"] + pairs["DD"]
        rand = 0.0 if denominator == 0.0 else numerator / denominator
        self.result[PyEvALLReport.AVERAGE_TAG] = rand

    def fire_preconditions(self, comparator):
        # Require complete non-overlapping clustering data.
        # Counting-pairs metrics require non-overlapping complete predictions.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        ids_gold = comparator.get_ids_gold()
        ids_pred = comparator.get_ids_pred()
        if len(ids_gold.difference(ids_pred)) > 0:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_PREDICTION_MISSING_GOLD_ITEMS, comparator.get_testcase())
            return True
        for item_id in ids_pred:
            if len(comparator.get_clusters_for_id_in_pred(item_id)) == 0:
                super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_PREDICTION_ITEMS_WITHOUT_CLUSTER, comparator.get_testcase())
                return True
        if comparator.has_items_in_multiple_clusters_gold():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True
        if comparator.has_items_in_multiple_clusters_pred():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True        
        return False


class Jaccard(PyEvALLMetric):
    """
    Jaccard index for positive clustering pairs.

    Formula:
        Jaccard = SS / (SS + SD + DS)

    Negative-negative agreements are intentionally ignored.
    """

    def __init__(self, evaluation_id):
        # Create the Jaccard clustering metric.
        super().__init__(MetricFactory.Jaccard.value, "Jaccard", "JACC", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute Jaccard over positive clustering pairs.
        self.log_evaluation_start(comparator, "Executing jaccard evaluation method")
        if self.fire_preconditions(comparator):
            return

        # Compute the positive-pair overlap between gold and predictions.
        pairs = comparator.get_pair_confusion_clustering()
        numerator = pairs["SS"]
        denominator = pairs["SS"] + pairs["SD"] + pairs["DS"]
        jaccard = 0.0 if denominator == 0.0 else numerator / denominator
        self.result[PyEvALLReport.AVERAGE_TAG] = jaccard

    def fire_preconditions(self, comparator):
        # Require complete non-overlapping clustering data with positive pairs.
        # Jaccard is not defined for overlapping clustering or absent positive pairs.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        ids_gold = comparator.get_ids_gold()
        ids_pred = comparator.get_ids_pred()
        if len(ids_gold.difference(ids_pred)) > 0:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_PREDICTION_MISSING_GOLD_ITEMS, comparator.get_testcase())
            return True
        for item_id in ids_pred:
            if len(comparator.get_clusters_for_id_in_pred(item_id)) == 0:
                super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_PREDICTION_ITEMS_WITHOUT_CLUSTER, comparator.get_testcase())
                return True
        if comparator.has_items_in_multiple_clusters_gold():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True
        if comparator.has_items_in_multiple_clusters_pred():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True
        pairs = comparator.get_pair_confusion_clustering()
        if pairs["SS"] + pairs["SD"] + pairs["DS"] == 0:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_NO_POSITIVE_PAIRS, comparator.get_testcase())
            return True
        return False


class FolkesAndMallows(PyEvALLMetric):
    """
    Fowlkes-Mallows index for clustering pairs.

    Formula:
        FM = sqrt((SS / (SS + SD)) * (SS / (SS + DS)))

    It is the geometric mean of pairwise precision and pairwise recall.
    """

    def __init__(self, evaluation_id):
        # Create the Fowlkes-Mallows clustering metric.
        super().__init__(MetricFactory.FolkesAndMallows.value, "Folkes and Mallows", "Folk", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute geometric mean of pairwise precision and recall.
        self.log_evaluation_start(comparator, "Executing folkes and mallows evaluation method")
        if self.fire_preconditions(comparator):
            return

        # Compute pairwise precision and recall from the pair confusion table.
        pairs = comparator.get_pair_confusion_clustering()
        numerator_sd = pairs["SS"]
        denominator_sd = pairs["SS"] + pairs["SD"]
        sd = 0.0 if denominator_sd == 0.0 else numerator_sd / denominator_sd

        numerator_ds = pairs["SS"]
        denominator_ds = pairs["SS"] + pairs["DS"]
        ds = 0.0 if denominator_ds == 0.0 else numerator_ds / denominator_ds

        multi = sd * ds
        fm = 0.0 if multi <= 0.0 else math.sqrt(multi)
        self.result[PyEvALLReport.AVERAGE_TAG] = fm

    def fire_preconditions(self, comparator):
        # Require complete non-overlapping clustering data with positive pairs.
        # Fowlkes-Mallows is not defined for overlapping clustering or absent positive pairs.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        ids_gold = comparator.get_ids_gold()
        ids_pred = comparator.get_ids_pred()
        if len(ids_gold.difference(ids_pred)) > 0:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_PREDICTION_MISSING_GOLD_ITEMS, comparator.get_testcase())
            return True
        for item_id in ids_pred:
            if len(comparator.get_clusters_for_id_in_pred(item_id)) == 0:
                super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_PREDICTION_ITEMS_WITHOUT_CLUSTER, comparator.get_testcase())
                return True
        if comparator.has_items_in_multiple_clusters_gold():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True
        if comparator.has_items_in_multiple_clusters_pred():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True
        pairs = comparator.get_pair_confusion_clustering()
        if pairs["SS"] + pairs["SD"] == 0 or pairs["SS"] + pairs["DS"] == 0:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_NO_POSITIVE_PAIRS, comparator.get_testcase())
            return True
        return False


class NMISG(PyEvALLMetric):
    """
    Strehl-Ghosh Normalized Mutual Information.

    Formula:
        NMI-SG(C, L) = MI(C, L) / sqrt(H(C) * H(L))
        MI(C, L) = sum_l,c P(l, c) * log2(P(l, c) / (P(l) * P(c)))

    It is evaluated only for non-overlapping clustering.
    """

    def __init__(self, evaluation_id):
        # Create the NMI-SG clustering metric.
        super().__init__(MetricFactory.NMISG.value, "Normalized Mutual Information SG", "NMI-SG", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute normalized mutual information for non-overlapping clusters.
        self.log_evaluation_start(comparator, "Executing normalized mutual information SG evaluation method")
        if self.fire_preconditions(comparator):
            return

        nmi = None
        ids_gold = comparator.get_ids_gold()
        ids_pred = comparator.get_ids_pred()
        total_instances = float(len(ids_gold))
        if total_instances != 0.0 and ids_gold.issubset(ids_pred):
            # Compute mutual information on the gold item universe.
            mutual_information = 0.0
            for cluster_gold in comparator.get_all_clusters_gold():
                ids_l = comparator.get_ids_for_cluster_in_gold(cluster_gold)
                size_l = float(len(ids_l))
                for cluster_pred in comparator.get_all_clusters_pred():
                    ids_c = comparator.get_ids_for_cluster_in_pred(cluster_pred).intersection(ids_gold)
                    size_c = float(len(ids_c))
                    intersection = float(len(ids_l.intersection(ids_c)))
                    denominator = size_l * size_c
                    if intersection != 0.0 and denominator != 0.0:
                        mutual_information += (intersection / total_instances) * math.log2((intersection * total_instances) / denominator)

            # Compute the entropy of gold classes.
            h_of_l = 0.0
            for cluster_gold in comparator.get_all_clusters_gold():
                p_l = len(comparator.get_ids_for_cluster_in_gold(cluster_gold)) / total_instances
                if p_l > 0.0:
                    h_of_l -= p_l * math.log2(p_l)

            # Compute the entropy of predicted clusters over the gold item universe.
            h_of_c = 0.0
            for cluster_pred in comparator.get_all_clusters_pred():
                p_c = len(comparator.get_ids_for_cluster_in_pred(cluster_pred).intersection(ids_gold)) / total_instances
                if p_c > 0.0:
                    h_of_c -= p_c * math.log2(p_c)

            denominator_nmi = math.sqrt(h_of_l * h_of_c)
            if denominator_nmi != 0.0:
                nmi = mutual_information / denominator_nmi

        self.result[PyEvALLReport.AVERAGE_TAG] = nmi

    def fire_preconditions(self, comparator):
        # Require non-overlapping clustering data.
        # NMI-SG is defined only for non-overlapping clustering-format comparators.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.has_items_in_multiple_clusters_gold():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True
        if comparator.has_items_in_multiple_clusters_pred():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True
        return False


class ReliabilityBCubed(PyEvALLMetric):
    """
    Extended BCubed precision, named reliability in the original implementation.

    Formula:
        Reliability = (1 / |occ_pred|) *
            sum_i sum_c in C(i) (1 / |c|) * sum_j in c mult_precision(i, j)
        mult_precision(i, j) = min(shared_pred(i, j), shared_gold(i, j)) / shared_pred(i, j)

    For each predicted item occurrence, the score averages multiplicity
    precision against the other items in the same predicted cluster.
    """

    def __init__(self, evaluation_id):
        # Create the Reliability BCubed metric.
        super().__init__(MetricFactory.ReliabilityBCubed.value, "Reliability", "R", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute extended BCubed precision over predicted memberships.
        self.log_evaluation_start(comparator, "Executing reliability bcubed evaluation method")
        if self.fire_preconditions(comparator):
            return

        reliability = 0.0
        # Only predicted items with a valid gold assignment participate.
        ids_pred = {
            item_id
            for item_id in comparator.get_ids_pred()
            if len(comparator.get_clusters_for_id_in_gold(item_id)) > 0
        }
        number_occurrences = float(sum(len(comparator.get_clusters_for_id_in_pred(item_id)) for item_id in ids_pred))

        for item_id in ids_pred:
            clusters_of_item = comparator.get_clusters_for_id_in_pred(item_id)
            sum_c = 0.0
            for cluster_name in clusters_of_item:
                # Compare item multiplicities with all valid items in the predicted cluster.
                ids_in_cluster = comparator.get_ids_for_cluster_in_pred(cluster_name).intersection(ids_pred)
                size_c = float(len(ids_in_cluster))
                sum_shared = 0.0
                duplicated_cluster_members = size_c > 1.0 and all(
                    len(comparator.get_clusters_for_id_in_pred(id_in_cluster)) > 1
                    for id_in_cluster in ids_in_cluster
                )
                for id_prime in ids_in_cluster:
                    if duplicated_cluster_members and id_prime == item_id:
                        continue
                    if len(comparator.get_clusters_for_id_in_gold(item_id)) > 0 and len(comparator.get_clusters_for_id_in_gold(id_prime)) > 0:
                        sum_shared += comparator.get_multiplicity_precision(item_id, id_prime)
                if size_c != 0.0:
                    sum_c += (1 / size_c) * sum_shared
            reliability += sum_c

        # Average over predicted item occurrences.
        if number_occurrences != 0.0:
            reliability = reliability * (1 / number_occurrences)

        self.result[PyEvALLReport.AVERAGE_TAG] = reliability

    def fire_preconditions(self, comparator):
        # Require a clustering comparator.
        # Extended BCubed accepts overlapping clustering.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        return False


class SensitivityBCubed(PyEvALLMetric):
    """
    Extended BCubed recall, named sensitivity in the original implementation.

    Formula:
        Sensitivity = (1 / |occ_gold|) *
            sum_i sum_l in L(i) (1 / |l|) * sum_j in l mult_recall(i, j)
        mult_recall(i, j) = min(shared_pred(i, j), shared_gold(i, j)) / shared_gold(i, j)

    For each gold item occurrence, the score averages multiplicity recall
    against the other items in the same gold class.
    """

    def __init__(self, evaluation_id):
        # Create the Sensitivity BCubed metric.
        super().__init__(MetricFactory.SensitivityBCubed.value, "Sensitivity", "S", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute extended BCubed recall over gold memberships.
        self.log_evaluation_start(comparator, "Executing sensitivity bcubed evaluation method")
        if self.fire_preconditions(comparator):
            return

        sensitivity = 0.0
        number_occurrences = float(comparator.get_num_cluster_occurrences_gold())
        ids_gold = comparator.get_ids_gold()

        for item_id in ids_gold:
            clusters_of_item = comparator.get_clusters_for_id_in_gold(item_id)
            sum_l = 0.0
            for cluster_name in clusters_of_item:
                # Compare item multiplicities with all items in the gold class.
                ids_in_cluster = comparator.get_ids_for_cluster_in_gold(cluster_name)
                size_l = float(len(ids_in_cluster))
                sum_shared = 0.0
                for id_prime in ids_in_cluster:
                    if len(comparator.get_clusters_for_id_in_pred(item_id)) > 0 and len(comparator.get_clusters_for_id_in_pred(id_prime)) > 0:
                        sum_shared += comparator.get_multiplicity_recall(item_id, id_prime)
                if size_l != 0.0:
                    sum_l += (1 / size_l) * sum_shared
            sensitivity += sum_l

        # Average over gold item occurrences.
        if number_occurrences != 0.0:
            sensitivity = sensitivity * (1 / number_occurrences)

        self.result[PyEvALLReport.AVERAGE_TAG] = sensitivity

    def fire_preconditions(self, comparator):
        # Require a clustering comparator.
        # Extended BCubed accepts overlapping clustering.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        return False


class Entropy(PyEvALLMetric):
    """
    Cluster entropy.

    Formula:
        Entropy(C, L) = - sum_c P(c) * sum_l P(c, l) * log2(P(c, l))

    It measures disorder of gold classes inside predicted clusters.
    """

    def __init__(self, evaluation_id):
        # Create the cluster Entropy metric.
        super().__init__(MetricFactory.Entropy.value, "Entropy", "E", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute entropy of gold classes inside predicted clusters.
        self.log_evaluation_start(comparator, "Executing entropy evaluation method")
        if self.fire_preconditions(comparator):
            return

        entropy = 0.0
        total_instances = float(comparator.get_num_instances_gold())
        if total_instances != 0.0:
            ids_gold = comparator.get_ids_gold()
            for cluster_pred in comparator.get_all_clusters_pred():
                # Ignore prediction-only items when measuring cluster disorder.
                ids_c = comparator.get_ids_for_cluster_in_pred(cluster_pred).intersection(ids_gold)
                size_c = len(ids_c)
                sum_p = 0.0
                for cluster_gold in comparator.get_all_clusters_gold():
                    intersection = len(comparator.get_ids_for_cluster_in_gold(cluster_gold).intersection(ids_c))
                    p_of_i_and_j = intersection / total_instances
                    if p_of_i_and_j > 0.0:
                        sum_p += p_of_i_and_j * math.log2(p_of_i_and_j)
                entropy += (size_c / total_instances) * sum_p
            entropy = entropy * -1

        self.result[PyEvALLReport.AVERAGE_TAG] = entropy

    def fire_preconditions(self, comparator):
        # Require complete non-overlapping clustering data.
        # Entropy requires complete, non-overlapping clustering assignments.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        ids_gold = comparator.get_ids_gold()
        ids_pred = comparator.get_ids_pred()
        if len(ids_gold.difference(ids_pred)) > 0:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_PREDICTION_MISSING_GOLD_ITEMS, comparator.get_testcase())
            return True
        for item_id in ids_pred:
            if len(comparator.get_clusters_for_id_in_pred(item_id)) == 0:
                super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_PREDICTION_ITEMS_WITHOUT_CLUSTER, comparator.get_testcase())
                return True
        if comparator.has_items_in_multiple_clusters_gold():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True
        if comparator.has_items_in_multiple_clusters_pred():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True
        return False


class ClassEntropy(PyEvALLMetric):
    """
    Class entropy.

    Formula:
        ClassEntropy(C, L) = - sum_l P(l) * sum_c P(c, l) * log2(P(c, l))

    It is the class-oriented counterpart of cluster entropy.
    """

    def __init__(self, evaluation_id):
        # Create the Class Entropy metric.
        super().__init__(MetricFactory.ClassEntropy.value, "Class Entropy", "CE", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute entropy of predicted clusters inside each gold class.
        self.log_evaluation_start(comparator, "Executing class entropy evaluation method")
        if self.fire_preconditions(comparator):
            return

        class_entropy = 0.0
        total_instances = float(comparator.get_num_instances_gold())
        if total_instances != 0.0:
            for cluster_gold in comparator.get_all_clusters_gold():
                # For each gold class, accumulate its distribution over predicted clusters.
                size_l = comparator.get_num_items_for_cluster_in_gold(cluster_gold)
                sum_p = 0.0
                for cluster_pred in comparator.get_all_clusters_pred():
                    intersection = comparator.get_cluster_intersection_size(cluster_gold, cluster_pred)
                    p_of_i_and_j = intersection / total_instances
                    if p_of_i_and_j > 0.0:
                        sum_p += p_of_i_and_j * math.log2(p_of_i_and_j)
                class_entropy += (size_l / total_instances) * sum_p
            class_entropy = class_entropy * -1

        self.result[PyEvALLReport.AVERAGE_TAG] = class_entropy

    def fire_preconditions(self, comparator):
        # Require complete non-overlapping clustering data.
        # Class entropy requires complete, non-overlapping clustering assignments.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        ids_gold = comparator.get_ids_gold()
        ids_pred = comparator.get_ids_pred()
        if len(ids_gold.difference(ids_pred)) > 0:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_PREDICTION_MISSING_GOLD_ITEMS, comparator.get_testcase())
            return True
        for item_id in ids_pred:
            if len(comparator.get_clusters_for_id_in_pred(item_id)) == 0:
                super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_CLUSTERING_PREDICTION_ITEMS_WITHOUT_CLUSTER, comparator.get_testcase())
                return True
        if comparator.has_items_in_multiple_clusters_gold():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True
        if comparator.has_items_in_multiple_clusters_pred():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER, comparator.get_testcase())
            return True
        return False


class VariationOfInformation(PyEvALLMetric):
    """
    Variation of Information.

    Formula:
        VI(C, L) = - sum_l,c |l intersect c| *
                   (log2(|l intersect c| / |l|) + log2(|l intersect c| / |c|))

    Lower values indicate more similar clusterings.
    """

    def __init__(self, evaluation_id):
        # Create the Variation of Information metric.
        super().__init__(MetricFactory.VariationOfInformation.value, "Variation of Information", "VI", evaluation_id)

    def evaluate(self, comparator, **params):
        # Compute variation of information between gold and predicted clusters.
        self.log_evaluation_start(comparator, "Executing variation of information evaluation method")
        if self.fire_preconditions(comparator):
            return

        vi = 0.0
        for cluster_gold in comparator.get_all_clusters_gold():
            size_l = comparator.get_num_items_for_cluster_in_gold(cluster_gold)
            for cluster_pred in comparator.get_all_clusters_pred():
                # Add the conditional information contribution for each non-empty intersection.
                size_c = comparator.get_num_items_for_cluster_in_pred(cluster_pred)
                intersection = comparator.get_cluster_intersection_size(cluster_gold, cluster_pred)
                if intersection != 0.0 and size_l != 0.0 and size_c != 0.0:
                    aux_l = intersection / size_l
                    aux_c = intersection / size_c
                    log_l = 0.0 if aux_l == 0.0 else math.log2(aux_l)
                    log_c = 0.0 if aux_c == 0.0 else math.log2(aux_c)
                    vi += intersection * (log_l + log_c)

        vi = vi * -1
        self.result[PyEvALLReport.AVERAGE_TAG] = vi

    def fire_preconditions(self, comparator):
        # Require a clustering comparator.
        # VI is defined for clustering-format comparators.
        if not comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLUSTERING]:
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        return False


        


