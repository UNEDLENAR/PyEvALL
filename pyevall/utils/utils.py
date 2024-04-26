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

class PyEvALLUtils(object):   
    dirname = os.path.dirname(__file__)
    LOG_FILENAME = os.path.join(dirname, 'file.conf' )
    PYEVALL_KEYS_TEXTS_REPORTS=os.path.join(dirname, 'pyevall_keys_texts_reports.rep' )
    MODULE_NAME="pyevall.metrics.metrics" 
    TEMP_FOLDER_EXECUTION= os.path.join(tempfile.gettempdir(), "pyevall_"+str(uuid.uuid4()))

    #PARAMS
    PARAM_HIERARCHY = "hierarchy"
    PARAM_FORMAT = "format" #options:"json", "tsv", "csv"
    PARAM_REPORT = "report" #options: "embedded", "dataframe"
    
    #OPTIONS PARAMS
    PARAM_OPTION_FORMAT_JSON= "json"
    PARAM_OPTION_FORMAT_TSV= "tsv"
    PARAM_OPTION_FORMAT_CSV= "csv"
    PARAM_OPTION_REPORT_EMBEDDED= "embedded"
    PARAM_OPTION_REPORT_DATAFRAME= "dataframe"
       
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




   
       
        