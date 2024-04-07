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
from pyevall.comparators.formats import PyEvALLFormat
from pyevall.utils.utils import PyEvALLUtils
import pandas as pd
import numpy as np
import logging.config

    
# Setting up the logger
logging.config.fileConfig(PyEvALLUtils.LOG_FILENAME, disable_existing_loggers=False)
logger = logging.getLogger(__name__)   
       
       
       
class Comparator(object):
    COMPARATOR_PROPERTY_CLASSIFICATION="classification"
    COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL="monolabel"
    COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL="multilabel"
    COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI="lewedi"
    COMPARATOR_PROPERTY_RANKING="ranking"


    def __init__(self, p_df, g_df, tc):
        logger.debug("Initializing object")            
        self.pred_df= p_df
        self.gold_df = g_df
        self.testcase=tc   
        self.proporties=({
            Comparator.COMPARATOR_PROPERTY_CLASSIFICATION:False,
            Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL:False,
            Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL:False,
            Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI:False,
            Comparator.COMPARATOR_PROPERTY_RANKING:False} 
        )    
            

    ##########################################
    #                                        #
    #     Functions for computing measures   #
    #                                        #
    ##########################################   
    def get_pred_df(self):
        return self.pred_df    
    
    
    def get_testcase(self):
        return self.testcase
    
    
    def get_classes_from_df(self, df):
        return df[PyEvALLFormat.VALUE].unique()    
    
    
    def get_classes_gold(self):
        if self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]:
            return self.gold_df[PyEvALLFormat.VALUE].unique()   
        elif self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]:
            return self.gold_df[PyEvALLFormat.VALUE].explode().unique()
    
    
    def get_col_as_list_df(self, df, col):        
        return df[col].tolist()        
    
    
    def filter_ids_by_gold(self, pred_df, gold_df):
        return pred_df[pred_df[PyEvALLFormat.TEST_CASE].isin(self.get_col_as_list_df(gold_df, PyEvALLFormat.ID))]
 
 
    def get_num_instances_gold(self):
        return self.gold_df.shape[0]     


    def get_num_instances_pred(self):
        return self.pred_df.shape[0]   
    
    
    def get_num_instances_gold_per_category_in_value(self, cl):
        if self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]:
            return self.gold_df[self.gold_df[PyEvALLFormat.VALUE] == cl].shape[0]  
        elif self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]:
            occurences = self.gold_df.apply(lambda row: self.is_category_in_array(cl, row[PyEvALLFormat.VALUE]), axis=1).tolist()
            return sum(occurences)   
    
    
    #Class may not exist in predictions
    def get_num_instances_pred_per_category_in_value(self, cl):        
        if len(self.pred_df)==0:
            return None
        if self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]:
            return self.pred_df[self.pred_df[PyEvALLFormat.VALUE] == cl].shape[0]     
        elif self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]:
            occurences = self.pred_df.apply(lambda row: self.is_category_in_array(cl, row[PyEvALLFormat.VALUE]), axis=1).tolist()
            return sum(occurences)
        
        
    def is_category_in_array(self, cl, classes):
        if cl in classes:
            return 1
        return 0
   
    
    #Check if there is only one class in both the gold and the output, and if there is the same number of instances.
    def is_1_category_in_value_goldAndPred_and_same_instances(self):   
        lst_gold = self.gold_df[PyEvALLFormat.VALUE].unique()
        lst_pred = self.pred_df[PyEvALLFormat.VALUE].unique()
        if len(lst_gold)==1 and len(lst_pred)==1:
            if lst_gold[0]==lst_pred[0]:
                lst_id_gold=self.gold_df[PyEvALLFormat.ID].unique()
                lst_id_pred=self.pred_df[PyEvALLFormat.ID].unique()
                dif1 = np.setdiff1d(lst_id_gold, lst_id_pred)
                dif2 = np.setdiff1d(lst_id_pred, lst_id_gold)
 
                difference = np.concatenate((dif1, dif2))
                if len(difference)==0:
                    return True
        return False    



           
class ClassificationComparator(Comparator):    
    def __init__(self, **params):               
        logger.debug("Initializing object")
        self.hierarchy = None       
       
        if PyEvALLUtils.PARAM_HIERARCHY in params:
            self.hierarchy= params[PyEvALLUtils.PARAM_HIERARCHY]

        self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION]=True
        self.preprocess_df_format_classification()
        self.conf_matrix_monolabel=dict()
        self.conf_matrix_multilabel=dict()
        self.index_classes=dict()
        
        
    def preprocess_df_format_classification(self):      
        lst_g_type=self.gold_df[PyEvALLFormat.VALUE].apply(type).unique()                  
        if lst_g_type[0]==type(""):
            self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]= True 
        elif lst_g_type[0]==type([]):
            self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]= True 
        elif lst_g_type[0]==type(dict()):
            self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI]= True
        else:
            self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION]= False 
            return                                  
 

    def generate_conf_matrix(self):
        logger.debug("Generating Confusion matrix")        
        self.generate_index_classes_and_init_matrix()  
        self.generate_conf_matrix_row()

            
    def generate_index_classes_and_init_matrix(self):
        if self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]:
            lst_classes=self.get_classes_gold()            
            for index, cl in enumerate(lst_classes):
                self.index_classes[cl]=index
                   
            size = len(self.index_classes)
            self.conf_matrix_monolabel = np.zeros(shape=(size, size))
            
        elif self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]:
            lst_classes = self.get_classes_gold()
            for cl in lst_classes:
                self.conf_matrix_multilabel[cl]=np.zeros(shape=(2, 2))    
    
    
    #Only ids that exist in gold are computed
    def generate_conf_matrix_row(self):                 
        self.gold_df.apply(lambda row: self.generate_pair_matrix(row[PyEvALLFormat.ID], row[PyEvALLFormat.VALUE],self.pred_df), axis=1)


    def generate_pair_matrix(self, gold_id, gold_value, pred_df):
        logger.debug("Adding pair id in confusion matrix: gold - (%s,%s) for testcase %s", gold_id, gold_value, self.testcase)
        row = pred_df[pred_df[PyEvALLFormat.ID] == gold_id]
        if self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]:
            if not row.empty:
                pred_value= row.iloc[0][PyEvALLFormat.VALUE]
                if pred_value in self.index_classes:
                    pos_gold = self.get_index_matrix_by_class(gold_value)
                    pos_pred = self.get_index_matrix_by_class(pred_value)
                    self.conf_matrix_monolabel[pos_gold, pos_pred]=self.conf_matrix_monolabel[pos_gold, pos_pred]+1
                    
        elif self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]:
            if not row.empty:
                pred_value= row.iloc[0][PyEvALLFormat.VALUE]                  
                for cl in self.conf_matrix_multilabel:                        
                    #True positive
                    if cl in gold_value and cl in pred_value:
                        self.conf_matrix_multilabel[cl][1][0]+=1
                    #False negative
                    elif cl in gold_value and not cl in pred_value:
                        self.conf_matrix_multilabel[cl][1][1]+=1
                    #True negative
                    elif not cl in gold_value and not cl in pred_value:
                        self.conf_matrix_multilabel[cl][0][0]+=1     
                    #False positive
                    elif not cl in gold_value and cl in pred_value:
                        self.conf_matrix_multilabel[cl][0][1]+=1            




    def get_index_matrix_by_class(self, cl):   
        logger.debug("Getting index of class %s in testcase %s", cl, self.testcase)
        #Generate confusion matrix if it is empty
        if len(self.conf_matrix_monolabel)==0 and len(self.conf_matrix_multilabel)==0:
            self.generate_conf_matrix()                    
        return self.index_classes[cl]


    def get_diagonal_conf_matrix(self):
        logger.debug("Getting diagonal for testcase %s", self.testcase)
        #Generate confusion matrix if it is empty
        if len(self.conf_matrix_monolabel)==0 and len(self.conf_matrix_multilabel)==0:
            self.generate_conf_matrix()    
        return self.conf_matrix_monolabel.diagonal()     
    
    
    def get_true_positive_per_class(self, cl):
        logger.debug("Getting true positive instances for class %s in testcase %s", cl, self.testcase)
        #Generate confusion matrix if it is empty
        if len(self.conf_matrix_monolabel)==0 and len(self.conf_matrix_multilabel)==0:
            self.generate_conf_matrix()
        
        if self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]:    
            index = self.get_index_matrix_by_class(cl)
            return self.conf_matrix_monolabel[index,index] 
           
        elif self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]:
            return self.conf_matrix_multilabel[cl][1][0]
            

    def get_pred_for_class(self,cl):
        logger.debug("Getting predicted instances for class %s in testcase %s", cl, self.testcase)
        #Generate confusion matrix if it is empty
        if len(self.conf_matrix_monolabel)==0 and len(self.conf_matrix_multilabel)==0:
            self.generate_conf_matrix()
            
        pred_for_class=0
        matrix = self.conf_matrix_monolabel
        index = self.get_index_matrix_by_class(cl)
        length = len(matrix)
        for i in range(length):
            pred_for_class+= matrix[i, index]
        
        return pred_for_class


        
      
class RankingComparator(Comparator):
    def __init__(self, **params):
        logger.debug("Initializing object")
        self.proporties[Comparator.COMPARATOR_PROPERTY_RANKING]=True
        self.preprocess_df_format_ranking()
        if self.proporties[Comparator.COMPARATOR_PROPERTY_RANKING]:
            self.pred_df_sorted= self.pred_df.sort_values(PyEvALLFormat.VALUE)


    def preprocess_df_format_ranking(self):         
        #Si el gold tiene diferentes tipos de datos en value es un error          
        lst_g_type=self.gold_df[PyEvALLFormat.VALUE].apply(type).unique()
        if not (lst_g_type[0]==type(1)):
            self.proporties[Comparator.COMPARATOR_PROPERTY_RANKING]= False
            return 
        
    
    def get_first_k_relevant_items_in_pred(self, param_k):
        relevants=0
        k= min(param_k, len(self.pred_df_sorted))
        for i in range(0,k):
            id = self.pred_df_sorted.iloc[i, 1]
            row = self.gold_df[self.gold_df[PyEvALLFormat.ID] == id]
            if len(row)>0:
                if row[PyEvALLFormat.VALUE].tolist()[0]>0:
                    relevants+=1
            
        return relevants
    
    
    def get_position_first_relevant_in_pred(self):
        k= len(self.pred_df_sorted)
        for i in range(0,k):
            id = self.pred_df_sorted.iloc[i, 1]
            row = self.gold_df[self.gold_df[PyEvALLFormat.ID] == id]
            if len(row)>0:
                if row[PyEvALLFormat.VALUE].tolist()[0]>0:
                    return (i+1)            
        return None
    
    
    def get_all_relevants_items_in_gold(self):
        return len(self.gold_df[self.gold_df[PyEvALLFormat.VALUE] >0])
    
    
    def get_list_ids_ordered_pred(self):
        return self.pred_df_sorted[PyEvALLFormat.ID].tolist()
    
    
    def get_list_perfect_rank_gold(self):
        return self.gold_df.sort_values(PyEvALLFormat.VALUE, ascending=False)[PyEvALLFormat.ID].tolist()
    
    
    def is_relevant_item_with_id_in_gold(self, id_pred):
        row = self.gold_df[self.gold_df[PyEvALLFormat.ID] == id_pred]
        if len(row)>0:
            if row[PyEvALLFormat.VALUE].tolist()[0]>0:
                return True
        return False
    
    
    def get_dict_values_gold_by_id(self):
        return  self.gold_df.set_index(PyEvALLFormat.ID)[PyEvALLFormat.VALUE].to_dict()
        

        
        
class PyEvALLComparator(ClassificationComparator, RankingComparator):
    def __init__(self, p_df, g_df, tc, **params):
        logger.debug("Initializing object")
        Comparator.__init__(self,p_df, g_df, tc)
        ClassificationComparator.__init__(self, **params)
        RankingComparator.__init__(self)


        
        
        
        