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
from enum import Enum
from pyevall.utils.utils import PyEvALLUtils
import importlib

    
class MetricFactory(Enum):  
    #CLASSIFICATION
    Accuracy = "Accuracy" 
    SystemPrecision = "SystemPrecision"
    Kappa = "Kappa"
    Precision = "Precision"
    Recall = "Recall"
    FMeasure = "FMeasure"    
    ICM = "ICM"
    ICMNorm = "ICMNorm"
    
    #CLASSIFICATION LeWiDi
    ICMSoft="ICMSoft"
    ICMSoftNorm = "ICMSoftNorm"
    CrossEntropy="CrossEntropy"
    MAE="MAE"
    
    #RANKING
    PrecisionAtK= "PrecisionAtK"
    RPrecision="RPrecision"
    MRR= "MRR"
    MAP= "MAP"
    DCG= "DCG"
    nDCG= "nDCG"
    ERR= "ERR"
    RBP = "RBP"
      
    
    @classmethod   
    def get_instance_metric(cls, metric, evaluation_id):
        logger = PyEvALLUtils.get_logger(__name__, evaluation_id)
        logger.debug("Generating instance of metric " + str(metric))
        instance=None
        try:
            module_ = importlib.import_module(str(PyEvALLUtils.MODULE_NAME))
            try:
                instance = getattr(module_, metric)(evaluation_id)
            except AttributeError:
                logger.debug("ERROR: The metric " + str(metric) + " does not exist.")
        except ImportError:
            logger.debug("ERROR: The module " + PyEvALLUtils.MODULE_NAME + " does not exist.")       
        return instance
    
    
    @classmethod 
    def get_lst_all_metrics(cls):
        lst_metrics=[]        
        #CLASSIFICATION
        lst_metrics.append(MetricFactory.Accuracy)
        lst_metrics.append(MetricFactory.SystemPrecision) 
        lst_metrics.append(MetricFactory.Kappa)        
        lst_metrics.append(MetricFactory.Precision)     
        lst_metrics.append(MetricFactory.Recall) 
        lst_metrics.append(MetricFactory.FMeasure) 
        lst_metrics.append(MetricFactory.ICM) 
        lst_metrics.append(MetricFactory.ICMNorm)          
    
        #CLASSIFICATION LeWeDi
        lst_metrics.append(MetricFactory.ICMSoft)     
        lst_metrics.append(MetricFactory.ICMSoftNorm)     
        lst_metrics.append(MetricFactory.CrossEntropy)  
        lst_metrics.append(MetricFactory.MAE)  
                    
        #RANKING
        lst_metrics.append(MetricFactory.PrecisionAtK)  
        lst_metrics.append(MetricFactory.RPrecision)  
        lst_metrics.append(MetricFactory.MRR)    
        lst_metrics.append(MetricFactory.MAP) 
        lst_metrics.append(MetricFactory.DCG) 
        lst_metrics.append(MetricFactory.nDCG) 
        lst_metrics.append(MetricFactory.ERR) 
        lst_metrics.append(MetricFactory.RBP) 
        return lst_metrics

        
    