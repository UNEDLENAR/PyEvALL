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
import jsbeautifier
import json
from pyevall.utils.utils import PyEvALLUtils
import pandas as pd
from tabulate import tabulate



class PyEvALLReport(object):
    """
    Represents a report generator for the PyEvALL evaluation tool.

    This class provides functionality to generate json reports containing file information, errors encountered and the results achieve with metrics
    during the evaluation process.
    """ 
       
    #TAGs class
    OK ="OK" 
    FAIL = "FAIL"
    WARNING = "WARNING"
    STOP = "STOP"
    CONTINUE= "CONTINUE"
    UNKNOWN="UNKNOWN"
    COMING_SOON="Coming soon!" 
    EMBEDDED_OPTION="Use parameter: report=\"embedded\"!"
    
    #TAGs main report
    METRIC_TAG="metrics"
    FILES_TAG="files"
    NAME_TAG="name"
    ACRONYM_TAG= "acronym"
    DESCRIPTION_TAG="description"
    STATUS_TAG="status"
    GOLD_TAG="gold"
    ERRORS_TAG="errors"
    WARNINGS_TAG="warnings"
    LINES_TAG="lines"
    EXCEPTION_TAG="exception"
    CLASSES_TAG= "classes"  
    RESULTS_TAG="results"   
    TEST_CASES_TAG="test_cases"
    AVERAGE_TAG= "average"
    AVERAGE_PER_TC_TAG = "average_per_test_case"
    PRECONDITIONS_TAG = "preconditions"
    
    #TAGS Metric errors and preconditions
    METRIC_UNKONW_METRIC_ERROR= "METRIC_UNKONW_METRIC_ERROR"
    METRIC_NOT_TEST_CASE_IN_COMMON_ERROR="METRIC_NOT_TEST_CASE_IN_COMMON_ERROR"
    METRIC_PRECONDITION_1_CLASS_GOLDANDPRED_AND_SAME_INSTANCES_ERROR="METRIC_PRECONDITION_1_CLASS_GOLDANDPRED_AND_SAME_INSTANCES_ERROR"
    METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION= "METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION"
    METRIC_PRECONDITION_NOT_IMPLEMENTED_EVALUATION_CONTEXT= "METRIC_PRECONDITION_NOT_IMPLEMENTED_EVALUATION_CONTEXT"
    METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC="METRIC_PRECONDITION_HIERARCHY_NOT_VALID_FOR_METRIC"
    METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING="METRIC_PRECONDITION_DUPLICATED_VALUES_RANKING"
    METRIC_PRECONDITION_EMPTY_PREDICTIONS_ERROR="METRIC_PRECONDITION_EMPTY_PREDICTIONS_ERROR"
    METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER="METRIC_PRECONDITION_NO_ITEMS_REPEATED_FOR_THE_SAME_CLUSTER"
    METRIC_PRECONDITION_CLUSTERING_PREDICTION_MISSING_GOLD_ITEMS="METRIC_PRECONDITION_CLUSTERING_PREDICTION_MISSING_GOLD_ITEMS"
    METRIC_PRECONDITION_CLUSTERING_PREDICTION_ITEMS_WITHOUT_CLUSTER="METRIC_PRECONDITION_CLUSTERING_PREDICTION_ITEMS_WITHOUT_CLUSTER"
    METRIC_PRECONDITION_CLUSTERING_NO_POSITIVE_PAIRS="METRIC_PRECONDITION_CLUSTERING_NO_POSITIVE_PAIRS"
    METRIC_PARAMS_NOT_DICT_WARNING="METRIC_PARAMS_NOT_DICT_WARNING"
    METRIC_PARAMS_FOR_METRIC_NOT_DICT_WARNING="METRIC_PARAMS_FOR_METRIC_NOT_DICT_WARNING"
    METRIC_PARAM_ALFA_NOT_VALID_WARNING="METRIC_PARAM_ALFA_NOT_VALID_WARNING"
    METRIC_PARAM_ALPHA_1_NOT_VALID_WARNING="METRIC_PARAM_ALPHA_1_NOT_VALID_WARNING"
    METRIC_PARAM_ALPHA_2_NOT_VALID_WARNING="METRIC_PARAM_ALPHA_2_NOT_VALID_WARNING"
    METRIC_PARAM_BETA_NOT_VALID_WARNING="METRIC_PARAM_BETA_NOT_VALID_WARNING"
    METRIC_PARAM_K_NOT_VALID_WARNING="METRIC_PARAM_K_NOT_VALID_WARNING"
    METRIC_PARAM_R_NOT_VALID_WARNING="METRIC_PARAM_R_NOT_VALID_WARNING"
    METRIC_PARAM_P_NOT_VALID_WARNING="METRIC_PARAM_P_NOT_VALID_WARNING"
        
    #TAGS Format parser errors
    FORMAT_FILE_NOT_EXIST_ERROR="FORMAT_FILE_NOT_EXIST_ERROR"
    FORMAT_EMPTY_FILE_ERROR="FORMAT_EMPTY_FILE_ERROR"
    

    #TAGS Format parser json
    FORMAT_INCORRECT_JSON_ERROR="FORMAT_INCORRECT_JSON_ERROR"
    FORMAT_INCORRECT_SCHEMA_JSON_ERROR="FORMAT_INCORRECT_SCHEMA_JSON_ERROR"
    
    #TAGS Format parser tsv and csv
    FORMAT_NO_HEADERS_ROW_ERROR="FORMAT_NO_HEADERS_ROW_ERROR"
    FORMAT_NUMBER_COLUMNS_ROW_ERROR="FORMAT_NUMBER_COLUMNS_ROW_ERROR"
    FORMAT_EMPTY_VALUE_ROW_ERROR="FORMAT_EMPTY_VALUE_ROW_ERROR"
    FORMAT_EMPTY_VALUE_ROW_WARNING="FORMAT_EMPTY_VALUE_ROW_WARNING"
    FORMAT_IDS_REPEATED_ROW_ERROR="FORMAT_IDS_REPEATED_ROW_ERROR"
    FORMAT_CLUSTER_EMPTY_TAG_IN_VALUE_ROW_ERROR="FORMAT_CLUSTER_EMPTY_TAG_IN_VALUE_ROW_ERROR"
    FORMAT_CLUSTER_DUPLICATED_TAG_IN_VALUE_ROW_ERROR="FORMAT_CLUSTER_DUPLICATED_TAG_IN_VALUE_ROW_ERROR"
    FORMAT_CSV_FORMAT_IDENTIFIED_WARNING="FORMAT_CSV_FORMAT_IDENTIFIED_WARNING"
    FORMAT_TSV_FORMAT_IDENTIFIED_WARNING="FORMAT_TSV_FORMAT_IDENTIFIED_WARNING"
    
    #TAGS Format generic error
    FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR="FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR"
    FORMAT_DIFFERENT_TYPES_IN_VALUE_GOLD_AND_PRED="FORMAT_DIFFERENT_TYPES_IN_VALUE_GOLD_AND_PRED"
    
    
    def __init__(self):
        # Initialize the report container and duplicate-file suffix index.
        self.report=dict()
        self.index=0
        

    def init_report(self):       
        # Create the top-level metric and file report sections.
        self.report[self.METRIC_TAG]=dict()
        self.report[self.FILES_TAG]=dict()
        
        
    ##########################################
    #                                        #
    #            BUILDING FILES REPORT       #
    #                                        #
    ##########################################    
    def insert_file(self, name, isGold):
        # Register an input file and initialize its parsing status.
        file=dict()
        file[self.NAME_TAG]=name
        file[self.STATUS_TAG]=self.OK
        file[self.GOLD_TAG]=isGold
        file[self.DESCRIPTION_TAG]=self.EMBEDDED_OPTION
        file[self.ERRORS_TAG]=dict()
        if not name in self.report[self.FILES_TAG]:
            self.report[self.FILES_TAG][name]=file
        else:
            extended_name=name+ "_duplicate_" + str(self.index)
            self.report[self.FILES_TAG][extended_name]=file
            self.index=self.index+1
        

    def insert_file_line_error(self, file_name, line, error, excep, stop_error):
        # Add a line-level file error and mark the file as failed.
        if file_name in self.report[self.FILES_TAG]:
            if not error in self.report[self.FILES_TAG][file_name][self.ERRORS_TAG]:
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error]=dict()
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.LINES_TAG]=[]
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.DESCRIPTION_TAG]=self.EMBEDDED_OPTION
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.EXCEPTION_TAG]=excep
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.STATUS_TAG]= self.STOP if stop_error==True else self.CONTINUE
            self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.LINES_TAG].append(line)    
            self.report[self.FILES_TAG][file_name][self.STATUS_TAG]=self.FAIL        
        else:
            raise Exception("Imposible error")

    def insert_file_line_warning(self, file_name, line, warning, excep, stop_error):
        # Add a line-level file warning without overriding a previous failure.
        if file_name in self.report[self.FILES_TAG]:
            if not warning in self.report[self.FILES_TAG][file_name][self.ERRORS_TAG]:
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][warning]=dict()
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][warning][self.LINES_TAG]=[]
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][warning][self.DESCRIPTION_TAG]=self.EMBEDDED_OPTION
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][warning][self.EXCEPTION_TAG]=excep
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][warning][self.STATUS_TAG]= self.STOP if stop_error==True else self.CONTINUE
            self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][warning][self.LINES_TAG].append(line)
            if self.report[self.FILES_TAG][file_name][self.STATUS_TAG] != self.FAIL:
                self.report[self.FILES_TAG][file_name][self.STATUS_TAG]=self.WARNING
        else:
            raise Exception("Imposible error")
            

    def insert_file_error(self, file_name, error, excep, stop_error):            
        # Add a file-level error and mark the file as failed.
        if file_name in self.report[self.FILES_TAG]:
            if not error in self.report[self.FILES_TAG][file_name][self.ERRORS_TAG]:
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error]=dict()
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.DESCRIPTION_TAG]=self.EMBEDDED_OPTION
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.EXCEPTION_TAG]=excep 
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.STATUS_TAG]= self.STOP if stop_error==True else self.CONTINUE
                self.report[self.FILES_TAG][file_name][self.STATUS_TAG]=self.FAIL          
        else:
            raise Exception("Imposible error")            
       

    def insert_file_testcase_error(self, file_name, test_case, error, excep, stop_error):
        # Add an error associated with a specific test case in a file.
        if file_name in self.report[self.FILES_TAG]:
            if not error in self.report[self.FILES_TAG][file_name][self.ERRORS_TAG]:
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error]=dict()
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.DESCRIPTION_TAG]=self.EMBEDDED_OPTION
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.EXCEPTION_TAG]=excep
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.TEST_CASES_TAG]=[]
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.STATUS_TAG]= self.STOP if stop_error==True else self.CONTINUE
            self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.TEST_CASES_TAG].append(test_case)    
            self.report[self.FILES_TAG][file_name][self.STATUS_TAG]=self.FAIL        
        else:
            raise Exception("Imposible error") 
    
    
    def insert_file_warning(self, file_name, warning, stop_error):            
        # Add a file-level warning and mark the file as warning if it exists.
        if file_name in self.report[self.FILES_TAG]:
            if not warning in self.report[self.FILES_TAG][file_name][self.ERRORS_TAG]:
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][warning]=dict()
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][warning][self.DESCRIPTION_TAG]=self.EMBEDDED_OPTION
                #self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][error][self.EXCEPTION_TAG]=excep 
                self.report[self.FILES_TAG][file_name][self.ERRORS_TAG][warning][self.STATUS_TAG]= self.STOP if stop_error==True else self.CONTINUE
                self.report[self.FILES_TAG][file_name][self.STATUS_TAG]=self.WARNING          
        else:
            raise Exception("Imposible error")  

        
    ##########################################
    #                                        #
    #            BUILDING METRICS REPORT     #
    #                                        #
    ########################################## 
    def init_metric(self, metric):
        # Register a metric and initialize its result section.
        self.report[self.METRIC_TAG][metric.class_name]=dict()
        self.report[self.METRIC_TAG][metric.class_name][self.NAME_TAG]=metric.name
        self.report[self.METRIC_TAG][metric.class_name][self.ACRONYM_TAG]=metric.acronym
        self.report[self.METRIC_TAG][metric.class_name][self.DESCRIPTION_TAG]=self.COMING_SOON
        self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG]=metric.status 
        self.report[self.METRIC_TAG][metric.class_name][self.RESULTS_TAG]= dict()
        self.report[self.METRIC_TAG][metric.class_name][self.RESULTS_TAG][self.TEST_CASES_TAG]= []     
    

    def insert_result_testcase_metric(self, metric, name_testcase):
        # Append one test-case result for a metric.
        testcase=dict()
        testcase[PyEvALLReport.NAME_TAG]=name_testcase  
        testcase.update(metric.result)  
        self.report[self.METRIC_TAG][metric.class_name][self.RESULTS_TAG][self.TEST_CASES_TAG].append(testcase)
        self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG]=metric.status 


    def insert_preconditions_metric(self, metric):
        # Store metric preconditions and propagate the metric status.
        if not self.PRECONDITIONS_TAG in self.report[self.METRIC_TAG][metric.class_name]:
            self.report[self.METRIC_TAG][metric.class_name][self.PRECONDITIONS_TAG]=dict()           
        for pred in metric.preconditions:
            self.report[self.METRIC_TAG][metric.class_name][self.PRECONDITIONS_TAG][pred]=metric.preconditions[pred]

        self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG]=metric.status 
        

    def insert_result_aveg_tc_metric(self, metric, aveg_tc):
        # Store the average over test cases for a metric.
        self.report[self.METRIC_TAG][metric.class_name][self.RESULTS_TAG][PyEvALLReport.AVERAGE_PER_TC_TAG]=aveg_tc         
    
  
    def insert_error_metric(self, metric, error):
        # Store an error for a known metric object.
        self.report[self.METRIC_TAG][metric.class_name]=dict()
        self.report[self.METRIC_TAG][metric.class_name][self.NAME_TAG]=metric.name    
        self.report[self.METRIC_TAG][metric.class_name][self.ACRONYM_TAG]=metric.acronym
        self.report[self.METRIC_TAG][metric.class_name][self.DESCRIPTION_TAG]=self.COMING_SOON   
        self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG]=self.FAIL
        if not self.ERRORS_TAG in self.report[self.METRIC_TAG][metric.class_name]:
            self.report[self.METRIC_TAG][metric.class_name][self.ERRORS_TAG]=[]
        
        error_report=dict()
        error_report[self.NAME_TAG]= error
        error_report[self.DESCRIPTION_TAG]=self.EMBEDDED_OPTION
        self.report[self.METRIC_TAG][metric.class_name][self.ERRORS_TAG].append(error_report)
        
    
    def insert_warning_metric(self, metric, warning):
        if not metric.class_name in self.report[self.METRIC_TAG]:
            self.report[self.METRIC_TAG][metric.class_name]=dict()
            self.report[self.METRIC_TAG][metric.class_name][self.NAME_TAG]=metric.name    
            self.report[self.METRIC_TAG][metric.class_name][self.ACRONYM_TAG]=metric.acronym
            self.report[self.METRIC_TAG][metric.class_name][self.DESCRIPTION_TAG]=self.COMING_SOON
            self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG]=metric.status
        if not self.WARNINGS_TAG in self.report[self.METRIC_TAG][metric.class_name]:
            self.report[self.METRIC_TAG][metric.class_name][self.WARNINGS_TAG]=[]
        
        warning_report=dict()
        warning_report[self.NAME_TAG]= warning
        warning_report[self.DESCRIPTION_TAG]=self.EMBEDDED_OPTION
        for current_warning in self.report[self.METRIC_TAG][metric.class_name][self.WARNINGS_TAG]:
            if current_warning[self.NAME_TAG]==warning:
                if self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG] != self.FAIL:
                    self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG]=self.WARNING
                return
        self.report[self.METRIC_TAG][metric.class_name][self.WARNINGS_TAG].append(warning_report)
        if self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG] != self.FAIL:
            self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG]=self.WARNING
        
    
    def insert_error_metric_unknown(self, metric, error):
        # Store an error for a metric name that is not known by the factory.
        self.report[self.METRIC_TAG][metric]=dict()
        self.report[self.METRIC_TAG][metric][self.NAME_TAG]=metric    
        self.report[self.METRIC_TAG][metric][self.ACRONYM_TAG]=self.UNKNOWN
        self.report[self.METRIC_TAG][metric][self.DESCRIPTION_TAG]=self.UNKNOWN    
        self.report[self.METRIC_TAG][metric][self.STATUS_TAG]=self.FAIL
        if not self.ERRORS_TAG in self.report[self.METRIC_TAG][metric]:
            self.report[self.METRIC_TAG][metric][self.ERRORS_TAG]=[]
        
        error_report=dict()
        error_report[self.NAME_TAG]= error
        error_report[self.DESCRIPTION_TAG]=self.EMBEDDED_OPTION
        self.report[self.METRIC_TAG][metric][self.ERRORS_TAG].append(error_report)
        
    def is_error_preconditions(self, preconditions):
        # Return True when any precondition is marked as failed.
        for pred in preconditions:
            precond=preconditions[pred]
            if PyEvALLReport.FAIL==precond[PyEvALLReport.STATUS_TAG]:
                return True        
    
    ##########################################
    #                                        #
    #           PRINT METHODS                #
    #                                        #
    ##########################################      
    def print_report(self):
        # Print the report using the representation provided by this class.
        options = jsbeautifier.default_options()
        options.indent_size = 2
        result= jsbeautifier.beautify(json.dumps(self.report,ensure_ascii=False), options)
        print(result)           




class PyEvALLEmbeddedReport(PyEvALLReport):
    """
    Represents an embedded report generator for the PyEvALL evaluation tool.

    This class inherits from PyEvALLReport and provides descriptive texts describing the errors found in the evaluation process embedding the descriptions 
    in the report.
    """
    FILE_CORRECTLY_PARSED="FILE_CORRECTLY_PARSED"
    FILE_INCORRECTLY_PARSED="FILE_INCORRECTLY_PARSED"
    EVALUATION_STATUS="EVALUATION_STATUS"
    FILE_NAME_DESCRIPTION="FILE_NAME_DESCRIPTION"
    TEST_CASE_DESCRIPTION="TEST_CASE_DESCRIPTION"
    LINES_DESCRIPTION="LINES_DESCRIPTION"
    METRIC_NAME_DESCRIPTION="METRIC_NAME_DESCRIPTION"
        
        
    def __init__(self, pyevall_report):
        # Reuse an existing raw report as the source for this view.
        self.report=pyevall_report.report 
        
        
    def generate_pyevall_embedded_report(self):   
        # Replace embedded placeholders with human-readable descriptions.
        self.parse_files_section(self.report[PyEvALLReport.FILES_TAG])  
        self.parse_metrics_section(self.report[PyEvALLReport.METRIC_TAG])
        
        
    def parse_metrics_section(self, metrics_section):
        # Enrich all metric sections.
        for metric in metrics_section:
            metric_object=metrics_section[metric]
            self.parse_metric_section(metric_object)
                
                
    def parse_metric_section(self, metric_section):
        # Extract and enrich errors or averages from one metric section.
        if PyEvALLReport.ERRORS_TAG in metric_section:
            errors = metric_section[PyEvALLReport.ERRORS_TAG]
            for error in errors:                
                self.add_description(error, error[PyEvALLReport.NAME_TAG])
                self.add_to_description_metric_name(error, metric_section[PyEvALLReport.NAME_TAG])
                self.add_to_description_status(metric_section)
        if PyEvALLReport.WARNINGS_TAG in metric_section:
            warnings = metric_section[PyEvALLReport.WARNINGS_TAG]
            for warning in warnings:
                self.add_description(warning, warning[PyEvALLReport.NAME_TAG])
                self.add_to_description_metric_name(warning, metric_section[PyEvALLReport.NAME_TAG])
                self.add_to_description_status(metric_section)
        if PyEvALLReport.PRECONDITIONS_TAG in metric_section:
            preconditions = metric_section[PyEvALLReport.PRECONDITIONS_TAG]
            for precondition in preconditions:
                prop = preconditions[precondition]           
                self.add_description(prop, precondition)
                self.add_to_description_metric_name(prop, metric_section[PyEvALLReport.NAME_TAG])
                self.add_to_description_test_case(prop)
                self.add_to_description_status(metric_section)
                        
        
    def parse_files_section(self, files_section):
        # Select the prediction file as the dataframe file label.
        for file in files_section:
            prop = files_section[file]
            if prop[PyEvALLReport.STATUS_TAG]==PyEvALLReport.OK:
                self.add_description(prop, self.FILE_CORRECTLY_PARSED)
                self.add_to_description_file_name(prop, prop[PyEvALLReport.NAME_TAG])
                continue
            else:          
                self.add_description(prop, self.FILE_INCORRECTLY_PARSED)
                self.add_to_description_file_name(prop, prop[PyEvALLReport.NAME_TAG])
                errors = prop[PyEvALLReport.ERRORS_TAG]
                for error in errors:
                    prop_1= errors[error]
                    self.add_description(prop_1, error)
                    self.add_to_description_file_name(prop_1, prop[PyEvALLReport.NAME_TAG])
                    
                    is_lines=False if not (PyEvALLReport.LINES_TAG in prop_1) else  len(prop_1[PyEvALLReport.LINES_TAG])>0                    
                    is_test_case=False if not (PyEvALLReport.TEST_CASES_TAG in prop_1) else  len(prop_1[PyEvALLReport.TEST_CASES_TAG])>0          
                    if not is_lines and is_test_case:                    
                        self.add_to_description_test_case(prop_1)
                    elif is_lines and not is_test_case:                     
                        self.add_to_description_lines(prop_1)    
                                          
                    self.add_to_description_status(prop_1)
            
        
    def add_description(self, prop, tag):   
        # Load the base description text for a report property.
        prop[PyEvALLReport.DESCRIPTION_TAG]=PyEvALLUtils.get_text_from_key(tag)
        
        
    def add_to_description_file_name(self, prop, name):   
        # Append the affected file name to a description.
        prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG]+ PyEvALLUtils.get_text_from_key(self.FILE_NAME_DESCRIPTION) + name + "."
        
        
    def add_to_description_status(self, prop):   
        # Append the evaluation status to a description.
        prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG] + PyEvALLUtils.get_text_from_key(self.EVALUATION_STATUS)+ prop[PyEvALLReport.STATUS_TAG]+ "."
    
    
    def add_to_description_test_case(self, prop):
        # Append the affected test cases to a description.
        prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG] + PyEvALLUtils.get_text_from_key(self.TEST_CASE_DESCRIPTION)
        for test in prop[PyEvALLReport.TEST_CASES_TAG]:
            prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG] + test+ ","   
        prop[PyEvALLReport.DESCRIPTION_TAG]=self.remove_last_character(prop[PyEvALLReport.DESCRIPTION_TAG]) +"."  
        
        
    def add_to_description_lines(self, prop):
        # Append the affected input lines to a description.
        prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG] + PyEvALLUtils.get_text_from_key(self.LINES_DESCRIPTION)
        for line in prop[PyEvALLReport.LINES_TAG]:
            prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG] + str(line) + ","   
        prop[PyEvALLReport.DESCRIPTION_TAG]=self.remove_last_character(prop[PyEvALLReport.DESCRIPTION_TAG]) +"."  
        
        
    def add_to_description_metric_name(self, prop, name):   
        # Append the metric name to a description.
        prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG]+ PyEvALLUtils.get_text_from_key(self.METRIC_NAME_DESCRIPTION) + name + "."
   
   
    def remove_last_character(self, str):
        # Remove the trailing separator added while composing descriptions.
        return str[:-1]        
   
   
     
        
class PyEvALLDataframeReport(PyEvALLReport):
    """
    Represents a dataframe report generator for the PyEvALL evaluation tool.

    This class inherits from PyEvALLReport and provides a different dataframes, according to different levels, with the results of the diffent metrics. 
    The object provide with 3 levels:
      * Results at average a the test_case level
      * Results at test_case level
      * Results at the class level, when the metric execute at the class level.
    """
    HEADER_TABLE_REPORT= "HEADER_TABLE_REPORT"

    
    def __init__(self, pyevall_report):
        # Reuse an existing raw report as the source for this view.
        if pyevall_report!=None:
            self.report=pyevall_report.report
        self.header= PyEvALLUtils.get_text_from_key(PyEvALLDataframeReport.HEADER_TABLE_REPORT)   
        self.df_average=None
        self.df_test_case=None
        self.df_test_case_classes=None

        
    def generate_pyevall_df_report(self):        
        # Generate average and test-case dataframe views.
        row = self.parse_files_section(self.report[PyEvALLReport.FILES_TAG])
        metrics_section = self.report[PyEvALLReport.METRIC_TAG]
        self.generate_df_average(row.copy(), metrics_section)
        self.generate_df_test_case(row.copy(), metrics_section)
 

    def parse_files_section(self, files_section):
        # Extract or enrich file-level information for this report view.
        for file in files_section:
            prop = files_section[file]
            if not prop[PyEvALLReport.GOLD_TAG]:
                return {PyEvALLReport.FILES_TAG: file}
        return {"-":"-"}     
    
    
    def generate_df_average(self, row, metrics_section):    
        # Build the one-row dataframe with metric averages.
        row.update(self.parse_metrics_section(metrics_section))
        self.df_average = pd.DataFrame(row, index=[0])   
   
    
    def parse_metrics_section(self, metrics_section):
        # Extract average metric values for the dataframe report.
        row = dict()
        for metric in metrics_section:
            metric_object=metrics_section[metric]
            acronym, result = self.parse_metric_section(metric_object)
            if result ==None:
                row [acronym]="-"
            else:
                row[acronym]=str(result)
        return row
            
                
    def parse_metric_section(self, metric_section):
        # Extract the acronym and average value from one metric section.
        acronym = None
        result= None
        if PyEvALLReport.ACRONYM_TAG in metric_section:
            acronym = metric_section[PyEvALLReport.ACRONYM_TAG]
        if PyEvALLReport.RESULTS_TAG in metric_section:
            if PyEvALLReport.AVERAGE_PER_TC_TAG in metric_section[PyEvALLReport.RESULTS_TAG]:
                result = metric_section[PyEvALLReport.RESULTS_TAG][PyEvALLReport.AVERAGE_PER_TC_TAG]
        return acronym, result

    
    def generate_df_test_case(self, row, metrics_section): 
        # Build dataframes with test-case and class-level metric values.
        df_dict, df_dict_classes= self.parse_metrics_section_for_tc(row[PyEvALLReport.FILES_TAG], metrics_section)  
        for tc in df_dict:
            if self.df_test_case==None:
                self.df_test_case = pd.DataFrame(df_dict[tc], index=[0])
            else:
                self.df_test_case = self.df_test_case.append(df_dict[tc], ignore_index = True) 
                
        if not len(df_dict_classes)==0 :
            for tc in df_dict_classes:
                if self.df_test_case_classes==None:
                    self.df_test_case_classes = pd.DataFrame(df_dict_classes[tc], index=[0])
                else:
                    self.df_test_case_classes = self.df_test_case_classes.append(df_dict_classes[tc], ignore_index = True) 
                  
    
    def parse_metrics_section_for_tc(self, file, metrics_section):
        # Extract test-case and class-level values from all metrics.
        df_dict = dict()
        df_dict_classes = dict()
        for metric in metrics_section:
            metric_object=metrics_section[metric]
            acronym = None
            if PyEvALLReport.ACRONYM_TAG in metric_object:
                acronym = metric_object[PyEvALLReport.ACRONYM_TAG]
            if PyEvALLReport.RESULTS_TAG in metric_object:
                results= metric_object[PyEvALLReport.RESULTS_TAG]
                if PyEvALLReport.TEST_CASES_TAG in results:
                    # Process test cases.
                    tc_table= self.parse_test_case_section(file, results[PyEvALLReport.TEST_CASES_TAG])
                    for tc_res in tc_table:
                        if not tc_res in df_dict:
                            df_dict[tc_res]= {PyEvALLDataframeReport.FILES_TAG: tc_res}
                        result=tc_table[tc_res]
                        if result ==None:
                            df_dict[tc_res][acronym]= "-"
                        else:
                            df_dict[tc_res][acronym]= tc_table[tc_res]
                            
                    # Process test cases with metrics at class level.
                    tc_classes_table = self.parse_test_case_classes_section(file, results[PyEvALLReport.TEST_CASES_TAG])
                    if not tc_classes_table==None:
                        for tc_res in tc_classes_table:
                            if not tc_res in df_dict_classes:
                                df_dict_classes[tc_res]= {PyEvALLDataframeReport.FILES_TAG: tc_res}
                                 
                            classes = tc_classes_table[tc_res]
                            for c in classes:
                                df_dict_classes[tc_res][acronym+"_"+c]= classes[c]                         
                                                 
        return df_dict, df_dict_classes     
    
    
    def parse_test_case_section(self, file, test_case):
        # Build the table of average values per test case.
        tc_table=dict()
        result = None
        for tc in test_case:
            if PyEvALLReport.NAME_TAG in tc:
                tc_name = file + "_"+ tc[PyEvALLReport.NAME_TAG]
                if PyEvALLReport.AVERAGE_TAG in tc:
                    result = tc[PyEvALLReport.AVERAGE_TAG]
                tc_table[tc_name] = result
        return tc_table
    
    
    def parse_test_case_classes_section(self, file, test_case):
        # Build the table of class-level values per test case.
        tc_table=dict()
        for tc in test_case:
            if not PyEvALLReport.CLASSES_TAG in tc:
                return None
            elif PyEvALLReport.NAME_TAG in tc:
                tc_name = file + "_"+ tc[PyEvALLReport.NAME_TAG]
                classes = tc[PyEvALLReport.CLASSES_TAG]
                tc_table[tc_name]=dict()
                for cl in classes:
                    tc_table[tc_name][cl]=classes[cl]
        return tc_table              
        

    def print_report(self):
        # Print dataframe views as PostgreSQL-style tables.
        print(self.header)
        if self.df_average is not None:
            print(tabulate(self.df_average, headers='keys', tablefmt='psql'))
        if self.df_test_case is not None:
            print(tabulate(self.df_test_case, headers='keys', tablefmt='psql'))
        if self.df_test_case_classes is not None:
            print(tabulate(self.df_test_case_classes.fillna("-"), headers='keys', tablefmt='psql'))        
    
    
    def print_report_to_markdown(self):
        # Print dataframe views as Markdown tables.
        print(self.header)
        if self.df_average is not None:
            print(self.df_average.to_markdown())
        if self.df_test_case is not None:
            print(self.df_test_case.to_markdown())
        if self.df_test_case_classes is not None:
            print(self.df_test_case_classes.fillna("-").to_markdown()) 
    
    
    def print_report_tsv(self):
        # Print dataframe views as TSV text.
        print(self.header)
        if self.df_average is not None:
            result = self.df_average.to_csv(index="True", sep="\t",
                       encoding="utf8",lineterminator='\n')
            print(result)
        
        if self.df_test_case is not None:
            result = self.df_test_case.to_csv(index="True", sep="\t",
                       encoding="utf8",lineterminator='\n')
            print(result) 
            
        if self.df_test_case_classes is not None:
            result = self.df_test_case_classes.fillna("-").to_csv(index="True", sep="\t",
                       encoding="utf8",lineterminator='\n')
            print(result) 



       
class PyEvALLMetaReport(PyEvALLReport):
    """Aggregates several raw PyEvALL reports under evaluation-specific keys."""
    EVALUATION_TAG="EVALUATION"    
        
          
    def add_pyevall_report(self, rep, i):  
        # Add one evaluation report to this aggregate report.
        self.report[PyEvALLMetaReport.EVALUATION_TAG + "_" + str(i)]=rep.report
        
        
        
        
class PyEvALLMetaReportDataFrame(PyEvALLDataframeReport):
    """Aggregates dataframe views generated from several PyEvALL reports."""

    def add_pyevall_report(self, rep, i):  
        # Add one evaluation dataframe report to this aggregate report.
        if i==1:
            df_report= PyEvALLDataframeReport(rep)
            df_report.generate_pyevall_df_report()
            self.df_average=df_report.df_average
            self.df_test_case=df_report.df_test_case
            self.df_test_case_classes= df_report.df_test_case_classes
        else:
            df_report = PyEvALLDataframeReport(rep)
            df_report.generate_pyevall_df_report()
            
            if self.df_average is not None:
                self.df_average = pd.concat([self.df_average,df_report.df_average], ignore_index=True)
            else:
                self.df_average=df_report.df_average
                
                
            if self.df_test_case is not None:
                self.df_test_case = pd.concat([self.df_test_case,df_report.df_test_case], ignore_index=True)
            else:
                self.df_test_case=df_report.df_test_case
                
                
            if self.df_test_case_classes is not None:
                self.df_test_case_classes = pd.concat([self.df_test_case_classes,df_report.df_test_case_classes], ignore_index=True)  
            else:
                self.df_test_case_classes= df_report.df_test_case_classes     
