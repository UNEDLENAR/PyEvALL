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
import jsbeautifier
import json
from pyevall.utils.utils import PyEvALLUtils
import pandas as pd
from tabulate import tabulate
import logging.config

# Setting up the logger
logging.config.fileConfig(PyEvALLUtils.LOG_FILENAME, disable_existing_loggers=False)
logger = logging.getLogger(__name__)



class PyEvALLReport(object):
    """
    Represents a report generator for the PyEvALL evaluation tool.

    This class provides functionality to generate json reports containing file information, errors encountered and the results achieve with metrics
    during the evaluation process.
    """ 
       
    #TAGs class
    OK ="OK" 
    FAIL = "FAIL"
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
    LINES_TAG="lines"
    EXCEPTION_TAG="exception"
    CLASSES_TAG= "classes"  
    RESULTS_TAG="results"   
    TEST_CASES_TAG="test_cases"
    AVERAGE_TAG= "average"
    AVERAGE_PER_TC_TAG = "average_per_test_case"
    
    #TAGS Metric errors and preconditions
    METRIC_UNKONW_METRIC_ERROR= "METRIC_UNKONW_METRIC_ERROR"
    PRECONDITIONS_TAG = "preconditions"
    METRIC_PRECONDITION_1_CLASS_GOLDANDPRED_AND_SAME_INSTANCES_ERROR="METRIC_PRECONDITION_1_CLASS_GOLDANDPRED_AND_SAME_INSTANCES_ERROR"
    METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION= "METRIC_PRECONDITION_NOT_VALID_FORMAT_FOR_CONTEXT_EVALUATION"
    METRIC_PRECONDITION_NOT_IMPLEMENTED_EVALUATION_CONTEXT= "METRIC_PRECONDITION_NOT_IMPLEMENTED_EVALUATION_CONTEXT"
        
    #TAGS Format parser errors
    FORMAT_FILE_NOT_EXIST_ERROR="FORMAT_FILE_NOT_EXIST_ERROR"
    FORMAT_EMPTY_FILE_ERROR="FORMAT_EMPTY_FILE_ERROR"

    #TAGS Format parser json
    FORMAT_INCORRECT_JSON_ERROR="FORMAT_INCORRECT_JSON_ERROR"
    FORMAT_INCORRECT_SCHEMA_JSON_ERROR="FORMAT_INCORRECT_SCHEMA_JSON_ERROR"
    
    #TAGS Format parser tsv
    FORMAT_NO_HEADERS_ROW_ERROR="FORMAT_NO_HEADERS_ROW_ERROR"
    FORMAT_NUMBER_COLUMNS_ROW_ERROR="FORMAT_NUMBER_COLUMNS_ROW_ERROR"
    FORMAT_EMPTY_VALUE_ROW_ERROR="FORMAT_EMPTY_VALUE_ROW_ERROR"
    FORMAT_IDS_REPEATED_ROW_ERROR="FORMAT_IDS_REPEATED_ROW_ERROR"
    
    #TAGS Format generic error
    FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR="FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR"
    FORMAT_DIFFERENT_TYPES_IN_VALUE_GOLD_AND_PRED="FORMAT_DIFFERENT_TYPES_IN_VALUE_GOLD_AND_PRED"
    
    
    def __init__(self):
        '''  '''
        self.report=dict()
        self.index=0
        

    def init_report(self):       
        self.report[self.METRIC_TAG]=dict()
        self.report[self.FILES_TAG]=dict()
        
        
    ##########################################
    #                                        #
    #            BUILDING FILES REPORT       #
    #                                        #
    ##########################################    
    def insert_file(self, name, isGold):
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
            

    def insert_file_error(self, file_name, error, excep, stop_error):            
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
    
        
    ##########################################
    #                                        #
    #            BUILDING METRICS REPORT     #
    #                                        #
    ########################################## 
    def init_metric(self, metric):
        self.report[self.METRIC_TAG][metric.class_name]=dict()
        self.report[self.METRIC_TAG][metric.class_name][self.NAME_TAG]=metric.name
        self.report[self.METRIC_TAG][metric.class_name][self.ACRONYM_TAG]=metric.acronym
        self.report[self.METRIC_TAG][metric.class_name][self.DESCRIPTION_TAG]=self.COMING_SOON
        self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG]=metric.status 
        self.report[self.METRIC_TAG][metric.class_name][self.RESULTS_TAG]= dict()
        self.report[self.METRIC_TAG][metric.class_name][self.RESULTS_TAG][self.TEST_CASES_TAG]= []     
    

    def insert_result_testcase_metric(self, metric, name_testcase):
        testcase=dict()
        testcase[PyEvALLReport.NAME_TAG]=name_testcase  
        testcase.update(metric.result)  
        self.report[self.METRIC_TAG][metric.class_name][self.RESULTS_TAG][self.TEST_CASES_TAG].append(testcase)
        self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG]=metric.status 


    def insert_preconditions_metric(self, metric):
        if not self.PRECONDITIONS_TAG in self.report[self.METRIC_TAG][metric.class_name]:
            self.report[self.METRIC_TAG][metric.class_name][self.PRECONDITIONS_TAG]=dict()           
        for pred in metric.preconditions:
            self.report[self.METRIC_TAG][metric.class_name][self.PRECONDITIONS_TAG][pred]=metric.preconditions[pred]

        self.report[self.METRIC_TAG][metric.class_name][self.STATUS_TAG]=metric.status 
        

    def insert_result_aveg_tc_metric(self, metric, aveg_tc):
        self.report[self.METRIC_TAG][metric.class_name][self.RESULTS_TAG][PyEvALLReport.AVERAGE_PER_TC_TAG]=aveg_tc         
    
  
    def insert_error_metric_unknown(self, metric, error):
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
    
    ##########################################
    #                                        #
    #           PRINT METHODS                #
    #                                        #
    ##########################################      
    def print_report(self):
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
        self.report=pyevall_report.report 
        
        
    def generate_pyevall_embedded_report(self):   
        self.parse_files_section(self.report[PyEvALLReport.FILES_TAG])  
        self.parse_metrics_section(self.report[PyEvALLReport.METRIC_TAG])
        
        
    def parse_metrics_section(self, metrics_section):
        for metric in metrics_section:
            metric_object=metrics_section[metric]
            self.parse_metric_section(metric_object)
                
                
    def parse_metric_section(self, metric_section):
        if PyEvALLReport.ERRORS_TAG in metric_section:
            errors = metric_section[PyEvALLReport.ERRORS_TAG]
            for error in errors:                
                self.add_description(error, error[PyEvALLReport.NAME_TAG])
                self.add_to_description_metric_name(error, metric_section[PyEvALLReport.NAME_TAG])
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
        prop[PyEvALLReport.DESCRIPTION_TAG]=PyEvALLUtils.get_text_from_key(tag)
        
        
    def add_to_description_file_name(self, prop, name):   
        prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG]+ PyEvALLUtils.get_text_from_key(self.FILE_NAME_DESCRIPTION) + name + "."
        
        
    def add_to_description_status(self, prop):   
        prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG] + PyEvALLUtils.get_text_from_key(self.EVALUATION_STATUS)+ prop[PyEvALLReport.STATUS_TAG]+ "."
    
    
    def add_to_description_test_case(self, prop):
        prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG] + PyEvALLUtils.get_text_from_key(self.TEST_CASE_DESCRIPTION)
        for test in prop[PyEvALLReport.TEST_CASES_TAG]:
            prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG] + test+ ","   
        prop[PyEvALLReport.DESCRIPTION_TAG]=self.remove_last_character(prop[PyEvALLReport.DESCRIPTION_TAG]) +"."  
        
        
    def add_to_description_lines(self, prop):
        prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG] + PyEvALLUtils.get_text_from_key(self.LINES_DESCRIPTION)
        for line in prop[PyEvALLReport.LINES_TAG]:
            prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG] + str(line) + ","   
        prop[PyEvALLReport.DESCRIPTION_TAG]=self.remove_last_character(prop[PyEvALLReport.DESCRIPTION_TAG]) +"."  
        
        
    def add_to_description_metric_name(self, prop, name):   
        prop[PyEvALLReport.DESCRIPTION_TAG]=prop[PyEvALLReport.DESCRIPTION_TAG]+ PyEvALLUtils.get_text_from_key(self.METRIC_NAME_DESCRIPTION) + name + "."
   
   
    def remove_last_character(self, str):
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
        if pyevall_report!=None:
            self.report=pyevall_report.report
        self.header= PyEvALLUtils.get_text_from_key(PyEvALLDataframeReport.HEADER_TABLE_REPORT)   
        self.df_average=None
        self.df_test_case=None
        self.df_test_case_classes=None

        
    def generate_pyevall_df_report(self):        
        row = self.parse_files_section(self.report[PyEvALLReport.FILES_TAG])
        metrics_section = self.report[PyEvALLReport.METRIC_TAG]
        self.generate_df_average(row.copy(), metrics_section)
        self.generate_df_test_case(row.copy(), metrics_section)
 

    def parse_files_section(self, files_section):
        for file in files_section:
            prop = files_section[file]
            if not prop[PyEvALLReport.GOLD_TAG]:
                return {PyEvALLReport.FILES_TAG: file}
        return {"-":"-"}     
    
    
    def generate_df_average(self, row, metrics_section):    
        row.update(self.parse_metrics_section(metrics_section))
        self.df_average = pd.DataFrame(row, index=[0])   
   
    
    def parse_metrics_section(self, metrics_section):
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
        acronym = None
        result= None
        if PyEvALLReport.ACRONYM_TAG in metric_section:
            acronym = metric_section[PyEvALLReport.ACRONYM_TAG]
        if PyEvALLReport.RESULTS_TAG in metric_section:
            if PyEvALLReport.AVERAGE_PER_TC_TAG in metric_section[PyEvALLReport.RESULTS_TAG]:
                result = metric_section[PyEvALLReport.RESULTS_TAG][PyEvALLReport.AVERAGE_PER_TC_TAG]
        return acronym, result

    
    def generate_df_test_case(self, row, metrics_section): 
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
                    #Process test cases
                    tc_table= self.parse_test_case_section(file, results[PyEvALLReport.TEST_CASES_TAG])
                    for tc_res in tc_table:
                        if not tc_res in df_dict:
                            df_dict[tc_res]= {PyEvALLDataframeReport.FILES_TAG: tc_res}
                        result=tc_table[tc_res]
                        if result ==None:
                            df_dict[tc_res][acronym]= "-"
                        else:
                            df_dict[tc_res][acronym]= tc_table[tc_res]
                            
                    #Process test cases with metric at the class level        
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
        print(self.header)
        print(tabulate(self.df_average, headers='keys', tablefmt='psql'))
        print(tabulate(self.df_test_case, headers='keys', tablefmt='psql'))
        print(tabulate(self.df_test_case_classes.fillna("-"), headers='keys', tablefmt='psql'))        
    
    
    def print_report_to_markdown(self):
        print(self.header)
        print(self.df_average.to_markdown())
        print(self.df_test_case.to_markdown())
        print(self.df_test_case_classes.fillna("-").to_markdown()) 
    
    
    def print_report_tsv(self):
        print(self.header)
        result = self.df_average.to_csv(index="True", sep="\t",
                       encoding="utf8",lineterminator='\n')
        print(result)
        
        result = self.df_test_case.to_csv(index="True", sep="\t",
                       encoding="utf8",lineterminator='\n')
        print(result) 
        
        result = self.df_test_case_classes.fillna("-").to_csv(index="True", sep="\t",
                       encoding="utf8",lineterminator='\n')
        print(result) 



       
class PyEvALLMetaReport(PyEvALLReport):
    EVALUATION_TAG="EVALUATION"    
        
          
    def add_pyevall_report(self, rep, i):  
        self.report[PyEvALLMetaReport.EVALUATION_TAG + "_" + str(i)]=rep.report
        
        
        
        
class PyEvALLMetaReportDataFrame(PyEvALLDataframeReport):             
    def add_pyevall_report(self, rep, i):  
            if i==0:
                df_report= PyEvALLDataframeReport(rep)
                df_report.generate_pyevall_df_report()
                self.df_average=df_report.df_average
                self.df_test_case=df_report.df_test_case
                self.df_test_case_classes= df_report.df_test_case_classes
            else:
                df_report = PyEvALLDataframeReport(rep)
                df_report.generate_pyevall_df_report()
                self.df_average = pd.concat([self.df_average,df_report.df_average], ignore_index=True)
                self.df_test_case = pd.concat([self.df_test_case,df_report.df_test_case], ignore_index=True)
                self.df_test_case_classes = pd.concat([self.df_test_case_classes,df_report.df_test_case_classes], ignore_index=True)       

        
        
        
        
                
        

        