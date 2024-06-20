# ============================================================================== 
#
# Copyright (c) 2022 - Permission is granted for use and modification of this file for research, non-commercial purposes. 
#
# This work has been financed by the European Union (NextGenerationEU funds) through the 
# ``Plan de Recuperación, Transformación y Resiliencia'', by the Ministry of Economic Affairs and Digital Transformation and by the UNED University. 
#
# The work has been developed within the project "ESPACIO DE OBSERVACIÓN DE INTELIGENCIA ARTIFICIAL (IA) EN ESPAÑOL”, in the framework of the 
# Convenio C039/21-OT between the public entity RED.ES, M.P. and the UNIVERSIDAD NACIONAL DE EDUCACIÓN A DISTANCIA (UNED). 
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
        self.evaluation_id= evaluation_id
        self.logger = PyEvALLUtils.get_logger(__name__, self.evaluation_id)
        self.class_name=class_name
        self.name= name
        self.acronym=acronym
        self.result=dict()
        self.preconditions=dict()
        self.status = PyEvALLReport.OK   
        
        
    def clear_results(self):
        self.result=dict()
        self.preconditions=dict()        
    
    
    def fire_preconditions(self, error, test_case):
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
    def evaluate(self):
        raise NotImplementedError("Please Implement this method")
    
    ############################################################################
    ##                                                                        ##
    ##                        CLASSIFICATION METRICS                          ##  
    ##                                                                        ##
    ############################################################################       
class Accuracy(PyEvALLMetric): 
    """
        @equation: Acc(p,g)= for each c in C sum(abs(intersection(Dc_p. Dc_g)))/ abs(Dg) where
                    
                    C represents the classes in the goldstandard.
                    Dc_p represents the set of hits for the class c in the predictions file
                    Dc_g represents the set of hits for the class c in the goldstandard file
                    Dg represents the number of items in the goldstandard file
                    
        @preconditions: None                
    """     
    def __init__(self, evaluation_id):        
        super().__init__(MetricFactory.Accuracy.value, "Accuracy", "Acc", evaluation_id)    

    def evaluate(self, comparator):  
        self.logger.info("Executing accuracy evaluation method")      
        
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return 
                
        sum_diagonal= comparator.get_diagonal_conf_matrix().sum()
        instances_gold= comparator.get_num_instances_gold()
        acc = sum_diagonal/instances_gold

        self.result[PyEvALLReport.AVERAGE_TAG]=acc
        self.logger.debug("Accuracy for the testcase %s is: %s", comparator.get_testcase(), acc)
            
    def fire_preconditions(self, comparator):
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False

        
                

class SystemPrecision(PyEvALLMetric):  
    """
        @equation: SP(p,g)= for each c in C sum(abs(intersection(Dc_p. Dc_g)))/ abs(Dp) where
                    
                    C represents the classes in the goldstandard.
                    Dc_p represents the set of hits for the class c in the predictions file
                    Dc_g represents the set of hits for the class c in the goldstandard file
                    Dp represents the number of items in the preditions  file
                    
        @preconditions: Predition and goldstandard files should contain different number of elements in all testcases               
    """     
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.SystemPrecision.value, "System Precision", "SP", evaluation_id)        


    def evaluate(self, comparator):   
        self.logger.info("Executing System Precision evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
                 
        sp= 0
        sum_diagonal= comparator.get_diagonal_conf_matrix().sum()
        instances_pred= comparator.get_num_instances_pred()
        if not instances_pred==None:
            sp = sum_diagonal/instances_pred

        self.result[PyEvALLReport.AVERAGE_TAG]=sp
        self.logger.debug("System Precision for the testcase %s is: %s", comparator.get_testcase(), sp)        
           
                
    def fire_preconditions(self, comparator):
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False   
                        
        
                
class Kappa(PyEvALLMetric):         
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.Kappa.value, "Cohen's Kappa", "Kappa", evaluation_id)           
        
        
    def evaluate(self, comparator):
        self.logger.info("Executing kappa evaluation method")
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
        error=False
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.is_1_category_in_value_goldAndPred_and_same_instances():
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_1_CLASS_GOLDANDPRED_AND_SAME_INSTANCES_ERROR, comparator.get_testcase())
            error=True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return error              
                



class Precision(PyEvALLMetric):        
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.Precision.value, "Precision", "Pr", evaluation_id)           
        
        
    def evaluate(self, comparator):
        self.logger.info("Executing precision evaluation method")
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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL] or
                     comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL])):  
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False     
                        
                
            
                    
class Recall(PyEvALLMetric):          
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.Recall.value, "Recall", "Re", evaluation_id)          
        
        
    def evaluate(self, comparator):
        self.logger.info("Executing recall evaluation method")
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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL] or
                     comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL])):   
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())              
        return False       
               
                
                
        
class FMeasure(PyEvALLMetric):     
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.FMeasure.value, "F-Measure", "F1", evaluation_id)  
        self.alfa_param=0.5          
                
                
    def evaluate(self, comparator):
        self.logger.info("Executing fmeasure evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return 

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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL] or
                     comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL])):  
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False

        


class ICM(PyEvALLMetric):   
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.ICM.value, "Information Contrast model", "ICM", evaluation_id)
        #parameters icm
        self.alpha_1=2
        self.alpha_2=2
        self.beta=3  
        
        #data structures for probabilities
        self.gold_freq= dict() 
        self.gold_prob= dict()      
        
        
    def generate_prob(self, comparator):
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
        gold_set=[]
        if np.isscalar(gold_row[PyEvALLFormat.VALUE]):
            gold_set.append(gold_row[PyEvALLFormat.VALUE])
        else:
            gold_set=gold_row[PyEvALLFormat.VALUE]  
            
        for c in gold_set:
            if not self.is_child(hierarchy, c):
                hierarchy[c]=[]     
    
    
    def calculate_prob_hierarchy_multi_labe(self, hierarchy,comparator):
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
    def evaluate(self, comparator):   
        self.logger.info("Executing ICM evaluation method")
        if self.fire_preconditions(comparator):
            return
        
        self.generate_prob(comparator) 
        
        result_icm = comparator.gold_df.apply(lambda row: self.calculate_icm_row(row,comparator), axis=1).tolist()

        gold_size = len(comparator.gold_df)
        average = sum(result_icm)/gold_size

        self.result[PyEvALLReport.AVERAGE_TAG]=average

        
    def fire_preconditions(self, comparator):
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
        size = len(classes)
        if size==0:
            return 0
        return self.get_prob_class(classes[0],comparator) + self.information_content(classes[1:size],comparator) - \
        self.information_content(self.calculate_set_deepest_common_ancestor( classes[0], classes[1:size], comparator), comparator)
                    
    
    def get_prob_class(self, clas,comparator):
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
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.ICMNorm.value, "Normalized Information Contrast Model", "ICM-Norm", evaluation_id)
       
    
    def evaluate(self, comparator):   
        self.logger.info("Executing ICM Normalized evaluation method")
        if self.fire_preconditions(comparator):
            return
        
        #Evaluate ICM pred vs gold
        icm = ICM(self.evaluation_id)
        icm.evaluate(comparator)
        res_pred= 0
        if PyEvALLReport.AVERAGE_TAG in icm.result:
            res_pred = icm.result[PyEvALLReport.AVERAGE_TAG] 
                   
        #Evaluate ICM gold vs gold  
        comp_gold = PyEvALLComparator(comparator.gold_df, comparator.gold_df, comparator.get_testcase(), self.evaluation_id)
        comp_gold.hierarchy= comparator.hierarchy
        icm = ICM(self.evaluation_id)
        icm.evaluate(comp_gold)
        res_gold=0
        if PyEvALLReport.AVERAGE_TAG in icm.result:
            res_gold=icm.result[PyEvALLReport.AVERAGE_TAG]
        
        #Calculate ICM Norm and truncate to 0 if the value is less than 0.
        icm_norm= (float(res_pred) - (res_gold*-1))/(res_gold-(res_gold*-1))
        if icm_norm<0:
            icm_norm=0

        self.result[PyEvALLReport.AVERAGE_TAG]=icm_norm

        
    def fire_preconditions(self, comparator):
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
    def __init__(self, evaluation_id):
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
        if comparator.hierarchy==None:
            comparator.gold_df[PyEvALLFormat.VALUE].apply(lambda value: self.search_classes(value))
        else:
            #check for classses not included in hierarchy   
            comparator.gold_df.apply(lambda row: self.check_class_not_in_hierachy(row, comparator.hierarchy), axis=1)            
            self.get_classes_hierarchy(comparator.hierarchy)
            
    
    def search_classes(self, value):
        gold_dict = value
        for c in gold_dict:
            if c not in self.lst_classes:
                self.lst_classes.append(c)  
                  
    
    def check_class_not_in_hierachy(self, gold_row, hierarchy):
        gold_set=[]
        if np.isscalar(gold_row[PyEvALLFormat.VALUE]):
            gold_set.append(gold_row[PyEvALLFormat.VALUE])
        else:
            gold_set=gold_row[PyEvALLFormat.VALUE]  
            
        for c in gold_set:
            if not self.is_child(hierarchy, c):
                hierarchy[c]=[]      
       
                        
    def is_child(self, hierarchy, child):
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
        if isinstance(hierarchy, dict):            
            for c in hierarchy: 
                self.lst_classes.append(c)             
                self.get_classes_hierarchy(hierarchy[c])  
        elif isinstance(hierarchy, list):
            for c in hierarchy:      
                self.lst_classes.append(c) 
                      
    
    #Method that calculate the propabilities for each class depending on the type of task.
    def calculate_probabilities(self, comparator):  
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
   
 
    #Method that propagates the weigth of classes between ancestors in hierarchical evaluations
    def propagate_max_weigth_ancestors(self, gold_dict, hierarchy, clas):
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
               
            
    #Method that expand the dataframe with all classes. Each column represent a class
    def expand_df(self, value):
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
    def evaluate(self, comparator):   
        self.logger.info("Executing ICM Soft evaluation method")
        if self.fire_preconditions(comparator):
            return
               
        self.get_list_classes(comparator)
        self.calculate_probabilities(comparator)
        
        result_icm_soft = comparator.gold_df.apply(lambda row: self.calculate_icm_row(row, comparator), axis=1).tolist()
        gold_size = len(comparator.gold_df)
        average = sum(result_icm_soft)/gold_size
        
        self.result[PyEvALLReport.AVERAGE_TAG]=average    

        
    def fire_preconditions(self, comparator):
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        return False     
    
        
    def calculate_icm_row(self, gold_row, comparator):
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
        size = len(classes)
        if size==0:
            return 0
        return self.get_prob_class(classes[0], comparator) + self.information_content(classes[1:size], comparator) - \
        self.information_content(self.calculate_set_deepest_common_ancestor(classes[0], classes[1:size], comparator), comparator)
                    
    
    def get_prob_class(self, tupla,comparator):
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
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.ICMSoftNorm.value, "Normalized Information Contrast Model Soft", "ICM-Soft-Norm", evaluation_id)      

   
    def evaluate(self, comparator):   
        self.logger.info("Executing ICM-Soft Normalized evaluation method")
        if self.fire_preconditions(comparator):
            return
        
        #Evaluate ICM pred vs gold
        icm_soft = ICMSoft(self.evaluation_id)
        icm_soft.evaluate(comparator)
        res_pred= 0
        if PyEvALLReport.AVERAGE_TAG in icm_soft.result:
            res_pred = icm_soft.result[PyEvALLReport.AVERAGE_TAG]        
            
        #Evaluate ICM gold vs gold       
        comp_gold = PyEvALLComparator(comparator.gold_df, comparator.gold_df, comparator.get_testcase(), self.evaluation_id)
        comp_gold.hierarchy= comparator.hierarchy
        icm_soft = ICMSoft(self.evaluation_id)
        icm_soft.evaluate(comp_gold)
        res_gold=0
        if PyEvALLReport.AVERAGE_TAG in icm_soft.result:
            res_gold=icm_soft.result[PyEvALLReport.AVERAGE_TAG]
        
        #Calculate ICM Norm and truncate to 0 if the value is less than 0.
        icm_norm= (float(res_pred) - (res_gold*-1))/(res_gold-(res_gold*-1))
        if icm_norm<0:
            icm_norm=0

        self.result[PyEvALLReport.AVERAGE_TAG]=icm_norm

        
    def fire_preconditions(self, comparator):
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        return False 





class CrossEntropy(PyEvALLMetric): 
    SMOOTH_VALUE=0.001
    
    
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.CrossEntropy.value, "Cross Entropy", "CE", evaluation_id)
       
       
    def smooth_and_normalize_data(self, gold_df, pred_df):           
        gold_df[PyEvALLFormat.VALUE] = gold_df[PyEvALLFormat.VALUE].apply(lambda row: self.normalize_data(self.smooth_data(row.copy())))   
        pred_df[PyEvALLFormat.VALUE] = pred_df[PyEvALLFormat.VALUE].apply(lambda row: self.normalize_data(self.smooth_data(row.copy())))
        
        
    def smooth_data(self, row):  
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
      
    
    def evaluate(self, comparator):
        self.logger.info("Executing Cross Entropy evaluation method")
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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())  
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False 


    def calculate_cross_entropy_instance(self, gold_row, pred_df):
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
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.MAE.value, "Mean Absolute Error", "MAE", evaluation_id)  
        self.lst_classes=[] 


    def get_list_classes(self, comparator):
        comparator.gold_df[PyEvALLFormat.VALUE].apply(lambda value: self.search_classes(value))

    def search_classes(self, value):
        gold_dict = value
        for c in gold_dict:
            if c not in self.lst_classes:
                self.lst_classes.append(c)  

    def expand_df(self, value):
        gold_dict = value
        new_columns=[]
        for c in self.lst_classes:
            if c in gold_dict:
                new_columns.append(gold_dict[c])
            else:
                new_columns.append(0.0)

        return pd.Series(new_columns, index=[self.lst_classes])  

    #We need to expand the dataframe
    def evaluate(self, comparator):   
        self.logger.info("Executing MAE evaluation method")
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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION] 
                and comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())  
            return True
        if comparator.hierarchy!=None:
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC, comparator.get_testcase())
        return False 


    def calculate_mae_instance(self, gold_row, pred_df, number_classes):    
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
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.PrecisionAtK.value, "Precision at k", "P@k", evaluation_id)    
        self.k_param=10     


    def evaluate(self, comparator):   
        self.logger.info("Executing Precision at K evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
                 
        p_at_k=None
        relevants = comparator.get_first_k_relevant_items_in_pred(self.k_param)
        if self.k_param!=0:
            p_at_k = relevants/self.k_param

        self.result[PyEvALLReport.AVERAGE_TAG]=p_at_k
        self.logger.debug("Precision at k for the testcase %s is: %s", comparator.get_testcase(), p_at_k)        
                
                
    def fire_preconditions(self, comparator):
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False     
    
    
    
    
class RPrecision(PyEvALLMetric):       
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.RPrecision.value, "R Precision", "RPre", evaluation_id)    


    def evaluate(self, comparator):   
        self.logger.info("Executing R Precision evaluation method")
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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False  
    
    
    
    
class MRR(PyEvALLMetric):    
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.MRR.value, "Mean Reciprocal Rank", "MRR", evaluation_id)    

    def evaluate(self, comparator):   
        self.logger.info("Executing Main Reciprocal Rank evaluation method")
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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]):  
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase())
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False      
    
    
    
    
class MAP(PyEvALLMetric):       
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.MAP.value, "Mean Average Precision", "MAP", evaluation_id) 
        self.r_param=1000   


    def evaluate(self, comparator):   
        self.logger.info("Executing MAP evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
        
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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase()) 
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False     
    
    
    
    
class DCG(PyEvALLMetric):     
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.DCG.value, "Discounted Cumulative Gain", "DCG", evaluation_id)   


    def evaluate(self, comparator):   
        self.logger.info("Executing DCG evaluation method")
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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase()) 
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False 
    
    
    
    
class nDCG(PyEvALLMetric):    
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.nDCG.value, "Normalized Discounted Cumulative Gain", "nDCG", evaluation_id)   


    def evaluate(self, comparator):   
        self.logger.info("Executing nDCG evaluation method")
        #Check preconditions of the metric
        if self.fire_preconditions(comparator):
            return   
        
        #Calculamos dcg sobre el ranking normal.
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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase()) 
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False    
        
    
    
class ERR(PyEvALLMetric):    
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.ERR.value, "Expected Reciprocal Rank", "ERR", evaluation_id)   


    def evaluate(self, comparator):   
        self.logger.info("Executing ERR evaluation method")
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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase()) 
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())  
        return False        
    
    
    
class RBP(PyEvALLMetric):    
    def __init__(self, evaluation_id):
        super().__init__(MetricFactory.RBP.value, "Rank Biased Precision", "RBP", evaluation_id)   
        self.p_param=0.8 


    def evaluate(self, comparator):   
        self.logger.info("Executing RBP evaluation method")
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
        if not (comparator.proporties[PyEvALLComparator.COMPARATOR_PROPERTY_RANKING]): 
            super().fire_preconditions(PyEvALLReport.METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION, comparator.get_testcase()) 
            return True
        if comparator.is_duplicate_values_true():
            super().fire_warning(PyEvALLReport.METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING, comparator.get_testcase())             
        return False 








        
