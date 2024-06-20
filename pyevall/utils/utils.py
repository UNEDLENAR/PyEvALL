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
import os
import tempfile
import uuid
import logging.config
from distutils.command.config import config

class PyEvALLUtils(object):   
    dirname = os.path.dirname(__file__)
    LOG_FILENAME = os.path.join(dirname, 'file.conf' )   
    PYEVALL_KEYS_TEXTS_REPORTS=os.path.join(dirname, 'pyevall_keys_texts_reports.rep' )
    MODULE_NAME="pyevall.metrics.metrics" 
    TEMP_FOLDER_EXECUTION= os.path.join(tempfile.gettempdir(), "pyevall_"+str(uuid.uuid4()))


    #PARAMS
    PARAM_HIERARCHY = "hierarchy"
    PARAM_REPORT = "report" #options: "embedded", "dataframe"
    PARAM_LOG_LEVEL="log_level"
    
    #OPTIONS PARAMS
    PARAM_OPTION_REPORT_SIMPLE= "simple"    
    PARAM_OPTION_REPORT_EMBEDDED= "embedded"
    PARAM_OPTION_REPORT_DATAFRAME= "dataframe"
    PARAM_OPTION_LOG_LEVEL_DEBUG="debug"
    PARAM_OPTION_LOG_LEVEL_INFO="info"
    PARAM_OPTION_LOG_LEVEL_NONE="none"
    CONFIGURATION=dict()
    #Default evaluation configuration
    #CONFIGURATION={
    #            PARAM_HIERARCHY:None,
    #            PARAM_REPORT:PARAM_OPTION_REPORT_SIMPLE,
    #            PARAM_LOG_LEVEL:PARAM_OPTION_LOG_LEVEL_INFO               
    #    }    
       
    #REPORTS
    keys_texts_reports = dict()

    FORMAT_JSON_SCHEMA= {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "test_case": {"type": "string"},
                "id":{"type": "string"},
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"},"minItems": 1},
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
      
    EvALLREPORT_JSON_SCHEM={    
        "type": "object",
        "properties":{
            "metrics":{
                "type": "object",
                "properties": {
                    "name": {"type": "string"}, 
                    "acronym": {"type": "string"}, 
                    "description": {"type": "string"}, 
                    "status":{"type": "string"}, 
                    "errors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties":{
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                    "results":{
                        "type": "object",
                        "properties": {
                            "test_cases": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "average":{"type": "number"},
                                        "classes": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "value": {"type": "number"},                                                 
                                            
                                                },
                                            },
                                        },
                                    },                              
                                },
                            },
                            "average_per_test_case": {"type": "number"}, 
                        },
                    },                    
                    "preconditions":{
                        "type": "object",
                        "properties": { 
                            "name": {"type": "string"}, 
                            "description": {"type": "string"}, 
                            "status": {"type": "string"},
                            "test_cases":{"type": "array", "items": {"type": "string"}},
                        },                        
                    },
                },
            },
            "files": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "status": {"type": "string"},
                    "gold": {"type": "boolean"},
                    "description": {"type": "string"},
                    "errors": {
                        "type": "object",
                        "properties": {
                            "lines": {"type": "string"},
                            "description": {"type": "string"}, 
                            "exception": {"type": "string"}, 
                            "test_case": {"type": "string"},   
                            "status":{"type": "string"},                                            
                        },                                    
                    },
                }, 
            }, 
        },             
    }
    
    @classmethod
    def load_configuration(cls, evaluation_id, **params):
        conf = PyEvALLUtils.get_active_configuration(evaluation_id)      
        
        if PyEvALLUtils.PARAM_HIERARCHY in params:
            conf[PyEvALLUtils.PARAM_HIERARCHY]=params[PyEvALLUtils.PARAM_HIERARCHY]        
        if PyEvALLUtils.PARAM_REPORT in params:
            conf[PyEvALLUtils.PARAM_REPORT]=params[PyEvALLUtils.PARAM_REPORT]                       
        if PyEvALLUtils.PARAM_LOG_LEVEL in params:
            conf[PyEvALLUtils.PARAM_LOG_LEVEL]=params[PyEvALLUtils.PARAM_LOG_LEVEL]
         
            
    @classmethod    
    def get_active_configuration(cls, evaluation_id):
        if not evaluation_id in PyEvALLUtils.CONFIGURATION:
            conf={
                        cls.PARAM_HIERARCHY:None,
                        cls.PARAM_REPORT:cls.PARAM_OPTION_REPORT_SIMPLE,
                        cls.PARAM_LOG_LEVEL:cls.PARAM_OPTION_LOG_LEVEL_INFO               
                } 
            PyEvALLUtils.CONFIGURATION[evaluation_id]= conf
        return PyEvALLUtils.CONFIGURATION[evaluation_id]             


    @classmethod
    def remove_active_configuration(cls, evaluation_id):
        if evaluation_id in cls.CONFIGURATION:
            cls.CONFIGURATION.pop(evaluation_id)
    
    
    
    @classmethod
    def get_logger(cls, name, evaluation_id):
        logging.config.fileConfig(PyEvALLUtils.LOG_FILENAME, disable_existing_loggers=False)        
        logger= logging.getLogger(name) 
        conf = PyEvALLUtils.get_active_configuration(evaluation_id)  
        if conf[cls.PARAM_LOG_LEVEL]==cls.PARAM_OPTION_LOG_LEVEL_NONE:
            logging.getLogger().removeHandler(logging.getLogger().handlers[0]) 
            hl= logging.NullHandler()
            logger.addHandler(hl)
        elif conf[cls.PARAM_LOG_LEVEL]==cls.PARAM_OPTION_LOG_LEVEL_DEBUG:
            logger.setLevel(logging.DEBUG)
        elif conf[cls.PARAM_LOG_LEVEL]==cls.PARAM_OPTION_LOG_LEVEL_INFO:
            logger.setLevel(logging.INFO)
        return logger  
    
    
    @classmethod
    def load_pair_texts(cls):
        with open(PyEvALLUtils.PYEVALL_KEYS_TEXTS_REPORTS) as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue
                key, value = line.rstrip("\n").split("=")
                if(not key in cls.keys_texts_reports):
                    cls.keys_texts_reports[key] = value
                else:
                    print("Duplicate assignment of key '%s'" % key)
    
    
    @classmethod
    def get_text_from_key(cls, key):
        if len(cls.keys_texts_reports)==0:
            cls.load_pair_texts()
            print("cargado " + str(len(cls.keys_texts_reports)))
        return cls.keys_texts_reports[key]
    
    
    @classmethod
    def get_tmp_dir_execution(cls):
        if not os.path.isdir(PyEvALLUtils.TEMP_FOLDER_EXECUTION):
            os.mkdir(PyEvALLUtils.TEMP_FOLDER_EXECUTION)
        return PyEvALLUtils.TEMP_FOLDER_EXECUTION




   
       
        