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
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.reports.reports import PyEvALLReport
from pyevall.utils.utils import PyEvALLUtils


def test_format_json():

    #Test Incorrect url
    test_format_json_incorrect_url_prediction()
    test_format_json_incorrect_url_gold()
    #Test empty file
    test_format_json_empty_prediction()
    test_format_json_empty_gold()    

    #Test incorrect json format
    test_format_json_incorrect_prediction()
    test_format_json_incorrect_gold()
    
    #Test incorrect schema
    test_format_json_incorrect_schema_prediction()
    test_format_json_incorrect_schema_gold()
    
    #Test duplicate id
    test_format_json_duplicates_ids_prediction()
    

    
def test_format_json_incorrect_url_prediction():
    eval = PyEvALLEvaluation()
    params={PyEvALLUtils.PARAM_FORMAT: PyEvALLUtils.PARAM_OPTION_FORMAT_JSON }
    m = []
    path="resources/format/json/"
    file_pred="asdf"
    file_gold="GOLD_MONO.txt"
    report_object = eval.evaluate(file_pred, path + file_gold, m, **params)
    report = report_object.report
    print("************** Testing json format: incorrect path predictions -- ", end=" ")
    for prop in report:
        if prop==PyEvALLReport.FILES_TAG:
            for error in report[prop][file_pred][PyEvALLReport.ERRORS_TAG]:
                if error==PyEvALLReport.FORMAT_FILE_NOT_EXIST_ERROR:
                    print("TEST PASSED, status error: ", report[prop][file_pred][PyEvALLReport.STATUS_TAG])
                else:
                    print("TEST FAILED")
            
            
def test_format_json_incorrect_url_gold():
    eval = PyEvALLEvaluation()
    params={PyEvALLUtils.PARAM_FORMAT: PyEvALLUtils.PARAM_OPTION_FORMAT_JSON }
    m = []
    path="resources/format/json/"
    file_pred="asdf"
    file_gold="sd"
    report_object = eval.evaluate(file_pred, path + file_gold, m, **params)
    report = report_object.report
    print("************** Testing json format: incorrect path gold -- ", end=" ")
    for prop in report:
        if prop==PyEvALLReport.FILES_TAG:
            for error in report[prop][file_gold][PyEvALLReport.ERRORS_TAG]:
                if error==PyEvALLReport.FORMAT_FILE_NOT_EXIST_ERROR:
                    print("TEST PASSED, status error: ", report[prop][file_gold][PyEvALLReport.STATUS_TAG])
                else:
                    print("TEST FAILED")
    

def test_format_json_empty_prediction():
    eval = PyEvALLEvaluation()
    params={PyEvALLUtils.PARAM_FORMAT: PyEvALLUtils.PARAM_OPTION_FORMAT_JSON }
    m = []
    path="resources/format/json/"
    file_pred="EMPTY"
    file_gold="GOLD_MONO.txt"
    report_object = eval.evaluate(path +file_pred, path + file_gold, m, **params)
    report = report_object.report
    print("************** Testing json format: empty predictions -- ", end=" ")
    for prop in report:
        if prop==PyEvALLReport.FILES_TAG:
            for error in report[prop][file_pred][PyEvALLReport.ERRORS_TAG]:
                if error==PyEvALLReport.FORMAT_EMPTY_FILE_ERROR:
                    print("TEST PASSED, status error: ",report[prop][file_pred][PyEvALLReport.STATUS_TAG])
                else:
                    print("TEST FAILED")                    
 
 
def test_format_json_empty_gold():
    eval = PyEvALLEvaluation()
    params={PyEvALLUtils.PARAM_FORMAT: PyEvALLUtils.PARAM_OPTION_FORMAT_JSON }
    m = []
    path="resources/format/json/"
    file_pred="EMPTY"
    file_gold="EMPTY"
    report_object = eval.evaluate(path +file_pred, path + file_gold, m, **params)
    report = report_object.report
    print("************** Testing json format: empty gold -- ", end=" ")
    for prop in report:
        if prop==PyEvALLReport.FILES_TAG:
            for error in report[prop][file_gold][PyEvALLReport.ERRORS_TAG]:
                if error==PyEvALLReport.FORMAT_EMPTY_FILE_ERROR:
                    print("TEST PASSED, status error: ",report[prop][file_gold][PyEvALLReport.STATUS_TAG])
                else:
                    print("TEST FAILED")                    
                    
                    
def test_format_json_incorrect_prediction():
    eval = PyEvALLEvaluation()
    params={PyEvALLUtils.PARAM_FORMAT: PyEvALLUtils.PARAM_OPTION_FORMAT_JSON }
    m = []
    path="resources/format/json/"
    file_pred="INCORRECT.json"
    file_gold="GOLD_MONO.json"
    report_object = eval.evaluate(path +file_pred, path + file_gold, m, **params)
    report = report_object.report
    print("************** Testing json format: incorrect json predictions -- ", end=" ")
    for prop in report:
        if prop==PyEvALLReport.FILES_TAG:
            for error in report[prop][file_pred][PyEvALLReport.ERRORS_TAG]:
                if error==PyEvALLReport.FORMAT_INCORRECT_JSON_ERROR:
                    print("TEST PASSED, status error: ",report[prop][file_pred][PyEvALLReport.STATUS_TAG])
                else:
                    print("TEST FAILED")   
                    
                    
def test_format_json_incorrect_gold():
    eval = PyEvALLEvaluation()
    params={PyEvALLUtils.PARAM_FORMAT: PyEvALLUtils.PARAM_OPTION_FORMAT_JSON }
    m = []
    path="resources/format/json/"
    file_pred="INCORRECT.json"
    file_gold="INCORRECT.json"
    report_object = eval.evaluate(path +file_pred, path + file_gold, m, **params)
    report = report_object.report
    print("************** Testing json format: incorrect json gold -- ", end=" ")
    for prop in report:
        if prop==PyEvALLReport.FILES_TAG:
            for error in report[prop][file_gold][PyEvALLReport.ERRORS_TAG]:
                if error==PyEvALLReport.FORMAT_INCORRECT_JSON_ERROR:
                    print("TEST PASSED, status error: ", report[prop][file_gold][PyEvALLReport.STATUS_TAG])  
                else:
                    print("TEST FAILED")                                    
                    
                    
def test_format_json_incorrect_schema_prediction():
    eval = PyEvALLEvaluation()
    params={PyEvALLUtils.PARAM_FORMAT: PyEvALLUtils.PARAM_OPTION_FORMAT_JSON }
    m = []
    path="resources/format/json/"
    file_pred="SCHEMA_INCORRECT.json"
    file_gold="GOLD_MONO.json"
    report_object = eval.evaluate(path +file_pred, path + file_gold, m, **params)
    report = report_object.report
    print("************** Testing json format: incorrect schema json predictions -- ", end=" ")

    for prop in report:
        if prop==PyEvALLReport.FILES_TAG:
            for error in report[prop][file_pred][PyEvALLReport.ERRORS_TAG]:
                if error==PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR:
                    print("TEST PASSED, status error: ", report[prop][file_pred][PyEvALLReport.STATUS_TAG]) 
                else:
                    print("TEST FAILED")   
    
    
def test_format_json_incorrect_schema_gold():
    eval = PyEvALLEvaluation()
    params={PyEvALLUtils.PARAM_FORMAT: PyEvALLUtils.PARAM_OPTION_FORMAT_JSON }
    m = []
    path="resources/format/json/"
    file_pred="SYS_MONO.json"
    file_gold="SCHEMA_INCORRECT.json"
    report_object = eval.evaluate(path +file_pred, path + file_gold, m, **params)
    report = report_object.report
    print("************** Testing json format: incorrect schema json predictions -- ", end=" ")
    for prop in report:
        if prop==PyEvALLReport.FILES_TAG:
            for error in report[prop][file_gold][PyEvALLReport.ERRORS_TAG]:
                if error==PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR:
                    print("TEST PASSED, status error: ", report[prop][file_gold][PyEvALLReport.STATUS_TAG])    
                else:
                    print("TEST FAILED")
                                        
                    
def test_format_json_duplicates_ids_prediction():
    eval = PyEvALLEvaluation()
    params={PyEvALLUtils.PARAM_FORMAT: PyEvALLUtils.PARAM_OPTION_FORMAT_JSON }
    m = []
    path="resources/format/json/"
    file_pred="SYS_DUPLICATE_IDS.json"
    file_gold="GOLD_MONO.json"
    report_object = eval.evaluate(path +file_pred, path + file_gold, m, **params)
    report = report_object.report
    print("************** Testing json format: duplicates ids predictions -- ", end=" ")
    for prop in report:
        if prop==PyEvALLReport.FILES_TAG:
            for error in report[prop][file_pred][PyEvALLReport.ERRORS_TAG]:
                if error==PyEvALLReport.FORMAT_IDS_REPEATED_ROW_ERROR:
                    print("TEST PASSED, status error: ", report[prop][file_pred][PyEvALLReport.STATUS_TAG])    
                else:
                    print("TEST FAILED")                 
    
 

if __name__ == '__main__':
    test_format_json()
    
    
    
    