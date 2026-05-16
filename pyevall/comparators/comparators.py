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
from pyevall.comparators.formats import PyEvALLFormat
from pyevall.utils.utils import PyEvALLUtils
import numpy as np
       
       
       
class Comparator(object):
    """
    Base comparator shared by all evaluation contexts.

    It stores the gold and prediction dataframes for one test case and exposes
    common helpers used by metrics, such as instance counts and class counts.
    Subclasses add context-specific preprocessing and operations.
    """
    COMPARATOR_PROPERTY_CLASSIFICATION="classification"
    COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL="monolabel"
    COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL="multilabel"
    COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI="lewedi"
    COMPARATOR_PROPERTY_RANKING="ranking"
    COMPARATOR_PROPERTY_CLUSTERING="clustering"


    def __init__(self, p_df, g_df, tc, evaluation_id):
        # Store the test-case dataframes and initialize context flags.
        self.evaluation_id=evaluation_id
        self.logger = PyEvALLUtils.get_logger(__name__, evaluation_id)    
        self.logger.debug("Initializing object")            
        self.pred_df= p_df
        self.gold_df = g_df
        self.testcase=tc   
        self.proporties=({
            Comparator.COMPARATOR_PROPERTY_CLASSIFICATION:False,
            Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL:False,
            Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL:False,
            Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_LEWEDI:False,
            Comparator.COMPARATOR_PROPERTY_RANKING:False,
            Comparator.COMPARATOR_PROPERTY_CLUSTERING:False} 
        )    
            

    ##########################################
    #                                        #
    #     Functions for computing measures   #
    #                                        #
    ##########################################   
    def get_pred_df(self):
        # Return the prediction dataframe for this test case.
        return self.pred_df    
    
    
    def get_testcase(self):
        # Return the name/id of the evaluated test case.
        return self.testcase
   
    
    def get_classes_gold(self):
        # Return gold classes according to the active classification format.
        if self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]:
            return self.gold_df[PyEvALLFormat.VALUE].unique()   
        elif self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]:
            return self.gold_df[PyEvALLFormat.VALUE].explode().unique()
    
    
    def get_col_as_list_df(self, df, col):        
        # Return one dataframe column as a plain Python list.
        return df[col].tolist()        
    
    
    def filter_ids_by_gold(self, pred_df, gold_df):
        # Keep only prediction rows whose identifiers are present in gold.
        return pred_df[pred_df[PyEvALLFormat.TEST_CASE].isin(self.get_col_as_list_df(gold_df, PyEvALLFormat.ID))]
 
 
    def get_num_instances_gold(self):
        # Return the number of gold rows for this test case.
        return self.gold_df.shape[0]     


    def get_num_instances_pred(self):
        # Return the number of prediction rows for this test case.
        return self.pred_df.shape[0]   
    
    
    def get_num_instances_gold_per_category_in_value(self, cl):
        # Count gold instances assigned to a class in mono-label or multi-label data.
        if self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]:
            return self.gold_df[self.gold_df[PyEvALLFormat.VALUE] == cl].shape[0]  
        elif self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]:
            occurences = self.gold_df.apply(lambda row: self.is_category_in_array(cl, row[PyEvALLFormat.VALUE]), axis=1).tolist()
            return sum(occurences)   
    
    
    #Class may not exist in predictions
    def get_num_instances_pred_per_category_in_value(self, cl):        
        # Count prediction instances assigned to a class, returning None for empty predictions.
        if len(self.pred_df)==0:
            return None
        if self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]:
            return self.pred_df[self.pred_df[PyEvALLFormat.VALUE] == cl].shape[0]     
        elif self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]:
            occurences = self.pred_df.apply(lambda row: self.is_category_in_array(cl, row[PyEvALLFormat.VALUE]), axis=1).tolist()
            return sum(occurences)
        
        
    def is_category_in_array(self, cl, classes):
        # Return 1 when a class is present in a multi-label value array.
        if cl in classes:
            return 1
        return 0
   
    
    #Check if there is only one class in both the gold and the output, and if there is the same number of instances.
    def is_1_category_in_value_goldAndPred_and_same_instances(self):   
        # Detect the degenerate case where gold and predictions contain the same single class and ids.
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
    """
    Comparator for classification contexts.

    It detects mono-label, multi-label, and LeWiDi inputs from the gold `value`
    type, builds confusion matrices for hard-label classification, and exposes
    per-class counts used by classification metrics.
    """
    def __init__(self):           
        # Mark classification context and prepare classification-specific state.
        self.logger.debug("Initializing object")
        self.hierarchy = PyEvALLUtils.get_active_configuration(self.evaluation_id)[PyEvALLUtils.PARAM_HIERARCHY]       

        self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION]=True
        self.preprocess_df_format_classification()
        self.conf_matrix_monolabel=dict()
        self.conf_matrix_multilabel=dict()
        self.index_classes=dict()
        
        
    def preprocess_df_format_classification(self):      
        # Identify whether classification values are mono-label, multi-label, or LeWiDi.
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
        # Build the classification confusion matrix for the current value format.
        self.logger.debug("Generating Confusion matrix")        
        self.generate_index_classes_and_init_matrix()  
        self.generate_conf_matrix_row()

            
    def generate_index_classes_and_init_matrix(self):
        # Create class indexes and initialize mono-label or multi-label matrices.
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
        # Fill confusion matrices by pairing each gold row with its prediction row.
        self.gold_df.apply(lambda row: self.generate_pair_matrix(row[PyEvALLFormat.ID], row[PyEvALLFormat.VALUE],self.pred_df), axis=1)


    def generate_pair_matrix(self, gold_id, gold_value, pred_df):
        # Add one gold/prediction item pair to the active confusion matrix.
        self.logger.debug("Adding pair id in confusion matrix: gold - (%s,%s) for testcase %s", gold_id, gold_value, self.testcase)
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
        # Return the matrix index assigned to a class, creating the matrix lazily.
        self.logger.debug("Getting index of class %s in testcase %s", cl, self.testcase)
        #Generate confusion matrix if it is empty
        if len(self.conf_matrix_monolabel)==0 and len(self.conf_matrix_multilabel)==0:
            self.generate_conf_matrix()                    
        return self.index_classes[cl]


    def get_diagonal_conf_matrix(self):
        # Return the true-positive diagonal of the mono-label confusion matrix.
        self.logger.debug("Getting diagonal for testcase %s", self.testcase)
        #Generate confusion matrix if it is empty
        if len(self.conf_matrix_monolabel)==0 and len(self.conf_matrix_multilabel)==0:
            self.generate_conf_matrix()    
        return self.conf_matrix_monolabel.diagonal()     
    
    
    def get_true_positive_per_class(self, cl):
        # Return true positives for one class in mono-label or multi-label data.
        self.logger.debug("Getting true positive instances for class %s in testcase %s", cl, self.testcase)
        #Generate confusion matrix if it is empty
        if len(self.conf_matrix_monolabel)==0 and len(self.conf_matrix_multilabel)==0:
            self.generate_conf_matrix()
        
        if self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MONOLABEL]:    
            index = self.get_index_matrix_by_class(cl)
            return self.conf_matrix_monolabel[index,index] 
           
        elif self.proporties[Comparator.COMPARATOR_PROPERTY_CLASSIFICATION_MULTILABEL]:
            return self.conf_matrix_multilabel[cl][1][0]
            

    def get_pred_for_class(self,cl):
        # Return the number of items predicted as a given class.
        self.logger.debug("Getting predicted instances for class %s in testcase %s", cl, self.testcase)
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
    """
    Comparator for ranking contexts.

    It validates numeric ranking values, keeps predictions ordered by rank, and
    provides relevance and rank helpers used by ranking metrics.
    """
    def __init__(self, **params):
        # Mark ranking context, validate ranking format, and cache sorted predictions.
        self.logger.debug("Initializing object")
        self.proporties[Comparator.COMPARATOR_PROPERTY_RANKING]=True
        self.preprocess_df_format_ranking()
        if self.proporties[Comparator.COMPARATOR_PROPERTY_RANKING]:
            self.pred_df_sorted= self.pred_df.sort_values(PyEvALLFormat.VALUE)
            self.duplicate_values=None


    def preprocess_df_format_ranking(self):         
        # Validate that ranking gold values are integer relevance scores.
        #If gold has different data types in value it is an error.          
        lst_g_type=self.gold_df[PyEvALLFormat.VALUE].apply(type).unique()
        if not (lst_g_type[0]==type(1)):
            self.proporties[Comparator.COMPARATOR_PROPERTY_RANKING]= False
            return 
        
    
    
    def get_first_k_relevant_items_in_pred(self, param_k):
        # Count relevant gold items retrieved in the first k predicted positions.
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
        # Return the one-based rank of the first relevant predicted item.
        k= len(self.pred_df_sorted)
        for i in range(0,k):
            id = self.pred_df_sorted.iloc[i, 1]
            row = self.gold_df[self.gold_df[PyEvALLFormat.ID] == id]
            if len(row)>0:
                if row[PyEvALLFormat.VALUE].tolist()[0]>0:
                    return (i+1)            
        return None
    
    
    def get_all_relevants_items_in_gold(self):
        # Count all relevant items in gold.
        return len(self.gold_df[self.gold_df[PyEvALLFormat.VALUE] >0])
    
    
    def get_list_ids_ordered_pred(self):
        # Return prediction ids ordered by predicted rank/value.
        return self.pred_df_sorted[PyEvALLFormat.ID].tolist()
    
    
    def get_list_perfect_rank_gold(self):
        # Return gold ids ordered by ideal descending relevance.
        return self.gold_df.sort_values(PyEvALLFormat.VALUE, ascending=False)[PyEvALLFormat.ID].tolist()
    
    
    def is_relevant_item_with_id_in_gold(self, id_pred):
        # Check whether an id is present in gold with positive relevance.
        row = self.gold_df[self.gold_df[PyEvALLFormat.ID] == id_pred]
        if len(row)>0:
            if row[PyEvALLFormat.VALUE].tolist()[0]>0:
                return True
        return False
    
    
    def get_dict_values_gold_by_id(self):
        # Return a dictionary mapping gold item ids to relevance values.
        return  self.gold_df.set_index(PyEvALLFormat.ID)[PyEvALLFormat.VALUE].to_dict()
        

    def is_duplicate_values_true(self):
        # Cache and return whether gold relevance values contain duplicates.
        if self.duplicate_values==None:
            self.duplicate_values= self.gold_df[PyEvALLFormat.VALUE].duplicated().any()
        return self.duplicate_values

        
class ClusteringComparator(Comparator):
    """
    Comparator for clustering contexts.

    It builds item-to-cluster and cluster-to-item indexes for gold and
    predictions, supports overlapping memberships, and provides pair,
    purity-style, entropy-style, and BCubed helper operations.
    """
    def __init__(self):
        # Mark clustering context and initialize lazy clustering indexes/caches.
        self.logger.debug("Initializing object")
        self.preprocess_df_format_clustering()
        self.item_to_clusters_gold = dict()
        self.item_to_clusters_pred = dict()
        self.clusters_to_items_gold = dict()
        self.clusters_to_items_pred = dict()
        self.common_ids = set()
        self._pair_confusion = None
        self._pair_labels_cache_gold = dict()
        self._pair_labels_cache_pred = dict()

    def preprocess_df_format_clustering(self):
        # Validate that clustering values are represented as cluster-name arrays.
        lst_g_type = self.gold_df[PyEvALLFormat.VALUE].apply(type).unique()
        if len(lst_g_type) == 1 and lst_g_type[0] == type([]):
            self.proporties[Comparator.COMPARATOR_PROPERTY_CLUSTERING] = True
        else:
            self.proporties[Comparator.COMPARATOR_PROPERTY_CLUSTERING] = False

    def _normalize_clusters(self, value):
        # Convert a cluster value into a set, using an empty set for invalid values.
        if not isinstance(value, list):
            return set()
        return set(value)

    def _ensure_clustering_data(self):
        # Lazily build item-to-cluster and cluster-to-item indexes for gold and predictions.
        if not self.proporties[Comparator.COMPARATOR_PROPERTY_CLUSTERING]:
            return
        if len(self.item_to_clusters_gold) != 0 or len(self.item_to_clusters_pred) != 0:
            return

        for _, row in self.gold_df.iterrows():
            item_id = row[PyEvALLFormat.ID]
            clusters = self._normalize_clusters(row[PyEvALLFormat.VALUE])
            self.item_to_clusters_gold[item_id] = clusters
            for cl in clusters:
                if cl not in self.clusters_to_items_gold:
                    self.clusters_to_items_gold[cl] = set()
                self.clusters_to_items_gold[cl].add(item_id)

        for _, row in self.pred_df.iterrows():
            item_id = row[PyEvALLFormat.ID]
            clusters = self._normalize_clusters(row[PyEvALLFormat.VALUE])
            self.item_to_clusters_pred[item_id] = clusters
            for cl in clusters:
                if cl not in self.clusters_to_items_pred:
                    self.clusters_to_items_pred[cl] = set()
                self.clusters_to_items_pred[cl].add(item_id)

        ids_gold = set(self.item_to_clusters_gold.keys())
        ids_pred = set(self.item_to_clusters_pred.keys())
        self.common_ids = ids_gold.intersection(ids_pred)

    def get_all_clusters_gold(self):
        # Return all cluster names found in the gold data.
        self._ensure_clustering_data()
        return list(self.clusters_to_items_gold.keys())

    def get_all_clusters_pred(self):
        # Return all cluster names found in the prediction data.
        self._ensure_clustering_data()
        return list(self.clusters_to_items_pred.keys())

    def get_ids_gold(self):
        # Return all item ids present in gold clustering data.
        self._ensure_clustering_data()
        return set(self.item_to_clusters_gold.keys())

    def get_ids_pred(self):
        # Return all item ids present in prediction clustering data.
        self._ensure_clustering_data()
        return set(self.item_to_clusters_pred.keys())

    def get_ids_for_cluster_in_gold(self, cluster_name):
        # Return item ids assigned to a gold cluster.
        self._ensure_clustering_data()
        return self.clusters_to_items_gold.get(cluster_name, set())

    def get_ids_for_cluster_in_pred(self, cluster_name):
        # Return item ids assigned to a predicted cluster.
        self._ensure_clustering_data()
        return self.clusters_to_items_pred.get(cluster_name, set())

    def get_clusters_for_id_in_gold(self, item_id):
        # Return gold clusters assigned to one item id.
        self._ensure_clustering_data()
        return self.item_to_clusters_gold.get(item_id, set())

    def get_clusters_for_id_in_pred(self, item_id):
        # Return predicted clusters assigned to one item id.
        self._ensure_clustering_data()
        return self.item_to_clusters_pred.get(item_id, set())

    def get_num_items_for_cluster_in_gold(self, cluster_name):
        # Return the number of items assigned to a gold cluster.
        return len(self.get_ids_for_cluster_in_gold(cluster_name))

    def get_num_items_for_cluster_in_pred(self, cluster_name):
        # Return the number of items assigned to a predicted cluster.
        return len(self.get_ids_for_cluster_in_pred(cluster_name))

    def get_valid_gold_cluster_items(self):
        # Return the set of items that belong to at least one gold cluster.
        self._ensure_clustering_data()
        valid_ids = set()
        for cluster_gold in self.get_all_clusters_gold():
            valid_ids.update(self.get_ids_for_cluster_in_gold(cluster_gold))
        return valid_ids

    def get_valid_pred_cluster_items(self, cluster_pred, valid_gold_ids=None):
        # Return predicted-cluster items that also exist in valid gold clustered items.
        self._ensure_clustering_data()
        if valid_gold_ids is None:
            valid_gold_ids = self.get_valid_gold_cluster_items()
        return self.get_ids_for_cluster_in_pred(cluster_pred).intersection(valid_gold_ids)

    def get_valid_pred_cluster_occurrences(self, clusters_pred=None, valid_gold_ids=None):
        # Count predicted cluster memberships restricted to valid gold items.
        self._ensure_clustering_data()
        if clusters_pred is None:
            clusters_pred = self.get_all_clusters_pred()
        if valid_gold_ids is None:
            valid_gold_ids = self.get_valid_gold_cluster_items()
        total = 0
        for cluster_pred in clusters_pred:
            total += len(self.get_valid_pred_cluster_items(cluster_pred, valid_gold_ids))
        return total

    def _pair_shares_cluster(self, cache, item_to_clusters, id_1, id_2):
        # Check and cache whether two items share at least one cluster.
        pair = (id_1, id_2) if id_1 <= id_2 else (id_2, id_1)
        if pair in cache:
            return cache[pair]
        c1 = item_to_clusters.get(id_1, set())
        c2 = item_to_clusters.get(id_2, set())
        result = len(c1.intersection(c2)) > 0
        cache[pair] = result
        return result

    def are_same_cluster_gold(self, id_1, id_2):
        # Return whether two items share a gold cluster.
        self._ensure_clustering_data()
        return self._pair_shares_cluster(self._pair_labels_cache_gold, self.item_to_clusters_gold, id_1, id_2)

    def are_same_cluster_pred(self, id_1, id_2):
        # Return whether two items share a predicted cluster.
        self._ensure_clustering_data()
        return self._pair_shares_cluster(self._pair_labels_cache_pred, self.item_to_clusters_pred, id_1, id_2)

    def get_pair_confusion_clustering(self):
        # Compute pair-count confusion values for clustering metrics.
        self._ensure_clustering_data()
        if self._pair_confusion is not None:
            return self._pair_confusion

        ids = list(self.common_ids)
        ss = 0
        sd = 0
        ds = 0
        dd = 0

        for i in range(0, len(ids)):
            for j in range(i + 1, len(ids)):
                id_1 = ids[i]
                id_2 = ids[j]
                same_gold = self.are_same_cluster_gold(id_1, id_2)
                same_pred = self.are_same_cluster_pred(id_1, id_2)
                if same_gold and same_pred:
                    ss += 1
                elif not same_gold and same_pred:
                    sd += 1
                elif same_gold and not same_pred:
                    ds += 1
                else:
                    dd += 1

        self._pair_confusion = {
            "SS": ss,
            "SD": sd,
            "DS": ds,
            "DD": dd,
            "TOTAL_PAIRS": ss + sd + ds + dd,
        }
        return self._pair_confusion

    def get_num_cluster_occurrences_gold(self):
        # Count total gold cluster memberships, including overlapping memberships.
        self._ensure_clustering_data()
        total = 0
        for item_id in self.item_to_clusters_gold:
            total += len(self.item_to_clusters_gold[item_id])
        return total

    def get_num_cluster_occurrences_pred(self):
        # Count total predicted cluster memberships, including overlapping memberships.
        self._ensure_clustering_data()
        total = 0
        for item_id in self.item_to_clusters_pred:
            total += len(self.item_to_clusters_pred[item_id])
        return total

    def has_items_in_multiple_clusters_gold(self):
        # Return whether any gold item belongs to more than one cluster.
        self._ensure_clustering_data()
        for item_id in self.item_to_clusters_gold:
            if len(self.item_to_clusters_gold[item_id]) > 1:
                return True
        return False

    def has_items_in_multiple_clusters_pred(self):
        # Return whether any predicted item belongs to more than one cluster.
        self._ensure_clustering_data()
        for item_id in self.item_to_clusters_pred:
            if len(self.item_to_clusters_pred[item_id]) > 1:
                return True
        return False

    def get_cluster_intersection_size(self, cluster_gold, cluster_pred):
        # Return the number of items shared by a gold cluster and a predicted cluster.
        ids_gold = self.get_ids_for_cluster_in_gold(cluster_gold)
        ids_pred = self.get_ids_for_cluster_in_pred(cluster_pred)
        return len(ids_gold.intersection(ids_pred))

    def get_precision_for_clusters(self, cluster_gold, cluster_pred):
        # Compute precision of a predicted cluster against a gold cluster.
        pred_size = self.get_num_items_for_cluster_in_pred(cluster_pred)
        if pred_size == 0:
            return 0.0
        return self.get_cluster_intersection_size(cluster_gold, cluster_pred) / pred_size

    def get_filtered_precision_for_clusters(self, cluster_gold, cluster_pred, valid_gold_ids=None):
        # Compute precision using only predicted items that are valid in gold.
        ids_pred = self.get_valid_pred_cluster_items(cluster_pred, valid_gold_ids)
        if len(ids_pred) == 0:
            return 0.0
        intersection = len(self.get_ids_for_cluster_in_gold(cluster_gold).intersection(ids_pred))
        return intersection / len(ids_pred)

    def get_recall_for_clusters(self, cluster_gold, cluster_pred):
        # Compute recall of a predicted cluster against a gold cluster.
        gold_size = self.get_num_items_for_cluster_in_gold(cluster_gold)
        if gold_size == 0:
            return 0.0
        return self.get_cluster_intersection_size(cluster_gold, cluster_pred) / gold_size

    def get_f_for_clusters(self, cluster_gold, cluster_pred):
        # Compute F-measure between one gold cluster and one predicted cluster.
        p = self.get_precision_for_clusters(cluster_gold, cluster_pred)
        r = self.get_recall_for_clusters(cluster_gold, cluster_pred)
        if p + r == 0:
            return 0.0
        return (2 * p * r) / (p + r)

    def get_filtered_f_for_clusters(self, cluster_gold, cluster_pred, valid_gold_ids=None):
        # Compute F-measure with precision restricted to valid predicted items.
        p = self.get_filtered_precision_for_clusters(cluster_gold, cluster_pred, valid_gold_ids)
        r = self.get_recall_for_clusters(cluster_gold, cluster_pred)
        if p + r == 0:
            return 0.0
        return (2 * p * r) / (p + r)

    def get_max_precision_for_pred_cluster(self, cluster_pred, clusters_gold=None):
        # Return the best precision obtained by a predicted cluster over gold clusters.
        if clusters_gold is None:
            clusters_gold = self.get_all_clusters_gold()
        max_precision = 0.0
        for cluster_gold in clusters_gold:
            max_precision = max(max_precision, self.get_precision_for_clusters(cluster_gold, cluster_pred))
        return max_precision

    def get_max_filtered_precision_for_pred_cluster(self, cluster_pred, clusters_gold=None, valid_gold_ids=None):
        # Return best filtered precision for a predicted cluster over gold clusters.
        if clusters_gold is None:
            clusters_gold = self.get_all_clusters_gold()
        max_precision = 0.0
        for cluster_gold in clusters_gold:
            max_precision = max(max_precision, self.get_filtered_precision_for_clusters(cluster_gold, cluster_pred, valid_gold_ids))
        return max_precision

    def get_max_precision_for_gold_cluster(self, cluster_gold, clusters_pred=None):
        # Return the best inverse-purity precision for a gold cluster over predicted clusters.
        if clusters_pred is None:
            clusters_pred = self.get_all_clusters_pred()
        max_precision = 0.0
        for cluster_pred in clusters_pred:
            # inverse-purity side must normalize by gold cluster size (Java behavior)
            max_precision = max(max_precision, self.get_recall_for_clusters(cluster_gold, cluster_pred))
        return max_precision

    def get_max_f_for_gold_cluster(self, cluster_gold, clusters_pred=None):
        # Return the best F-measure for a gold cluster over predicted clusters.
        if clusters_pred is None:
            clusters_pred = self.get_all_clusters_pred()
        max_f = 0.0
        for cluster_pred in clusters_pred:
            max_f = max(max_f, self.get_f_for_clusters(cluster_gold, cluster_pred))
        return max_f

    def get_max_filtered_f_for_gold_cluster(self, cluster_gold, clusters_pred=None, valid_gold_ids=None):
        # Return the best filtered F-measure for a gold cluster over predicted clusters.
        if clusters_pred is None:
            clusters_pred = self.get_all_clusters_pred()
        max_f = 0.0
        for cluster_pred in clusters_pred:
            max_f = max(max_f, self.get_filtered_f_for_clusters(cluster_gold, cluster_pred, valid_gold_ids))
        return max_f

    def get_num_shared_clusters_gold(self, id_1, id_2):
        # Count how many gold clusters are shared by two items.
        self._ensure_clustering_data()
        c1 = self.item_to_clusters_gold.get(id_1, set())
        c2 = self.item_to_clusters_gold.get(id_2, set())
        return len(c1.intersection(c2))

    def get_num_shared_clusters_pred(self, id_1, id_2):
        # Count how many predicted clusters are shared by two items.
        self._ensure_clustering_data()
        c1 = self.item_to_clusters_pred.get(id_1, set())
        c2 = self.item_to_clusters_pred.get(id_2, set())
        return len(c1.intersection(c2))

    def get_multiplicity_precision(self, id_1, id_2):
        # Compute pair multiplicity precision for overlapping clustering.
        inter_pred = self.get_num_shared_clusters_pred(id_1, id_2)
        inter_gold = self.get_num_shared_clusters_gold(id_1, id_2)
        numerator = min(inter_pred, inter_gold)
        if inter_pred == 0:
            return 0.0
        return numerator / inter_pred

    def get_multiplicity_recall(self, id_1, id_2):
        # Compute pair multiplicity recall for overlapping clustering.
        inter_pred = self.get_num_shared_clusters_pred(id_1, id_2)
        inter_gold = self.get_num_shared_clusters_gold(id_1, id_2)
        numerator = min(inter_pred, inter_gold)
        if inter_gold == 0:
            return 0.0
        return numerator / inter_gold
        
        
class PyEvALLComparator(ClassificationComparator, RankingComparator, ClusteringComparator):
    """
    Combined comparator for one evaluated test case.

    The comparator initializes every context-specific mixin. Each mixin marks
    whether its context is valid, allowing metrics to decide through
    preconditions whether they can be evaluated.
    """
    def __init__(self, p_df, g_df, tc, evaluation_id):
        # Initialize all comparator contexts for a single test case.
        Comparator.__init__(self,p_df, g_df, tc, evaluation_id)
        ClassificationComparator.__init__(self)
        RankingComparator.__init__(self)
        ClusteringComparator.__init__(self)


        
        
        
        
