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
import pyevall.comparators.comparators as comparators
from pyevall.reports.reports import PyEvALLReport
from pyevall.utils.utils import PyEvALLUtils
import pandas as pd
import csv
import os
import pathlib as p
import json
import jsonschema
from jsonschema import validate


class PyEvALLFormat(object):  
    DELIMITER_TSV = "\t"
    DELIMITER_CSV = ","
    TEST_CASE = "test_case"
    ID = "id"
    VALUE = "value" 
    

    def __init__(self, pyevall_report, pred_file, gold_file, evaluation_id): 
        """
        Constructor for initializing an instance of the class.
        
        Parameters:
            - pyevall_report: An instance of PyEvALLReport class to manage the evaluation report.
            - pred_file: Path to the file containing system predictions.
            - gold_file: Path to the file containing the gold standard labels.

            
        Functionality:
            - Initializes attributes and configures the evaluation based on the provided files and parameters.
            - Determines the format of input files and converts them to JSON format if needed.
            - Parses the JSON-formatted files for further processing.
            - If files or formats are invalid, sets valid_execution flag to False.
            - Logs debugging information during the initialization process.
            - Returns an initialized instance ready for evaluation.
        """    
        
        self.evaluation_id = evaluation_id         
        self.logger = PyEvALLUtils.get_logger(__name__, evaluation_id)
        #atributtes
        self.valid_execution=True
        self.pyevall_report=pyevall_report
       
        self.pred_path=pred_file
        self.pred_file_name= os.path.split(self.pred_path)[1]       
        self.pred_df=dict()
        self.pyevall_report.insert_file(self.pred_file_name, False)
        
        self.gold_path=gold_file
        self.gold_file_name= os.path.split(self.gold_path)[1]
        self.gold_df=dict()
        self.pyevall_report.insert_file(self.gold_file_name, True)
                
        #check if the predictions file exist 
        if self.check_file_exist(self.pred_path, self.pred_file_name):
            #if file does not exist we can not evaluate
            if self.check_file_exist(self.gold_path, self.gold_file_name):
                #Identify format, convert to json and parse it
                self.logger.debug("Initializing PyEvALLFormat object")   
                update_path, path= self.check_valids_for_tsv_csv_formats(self.pred_path, self.pred_file_name)
                if update_path:
                    self.pred_path=path
                update_path, path= self.check_valids_for_tsv_csv_formats(self.gold_path, self.gold_file_name)
                if update_path:
                    self.gold_path=path
                     
                self.parse_files_json_format()
            else:
                self.valid_execution=False                                   
                
        else:
            #The prediction file contains errors so the execution is not valid, but we parser the gold to detect possible errors 
            self.valid_execution=False
            #Check gold for errors
            if self.check_file_exist(self.gold_path, self.gold_file_name):
                update_path, path= self.check_valids_for_tsv_csv_formats(self.gold_path, self.gold_file_name)
                if update_path:
                    self.gold_path=path
                    
                self.parse_files_json_format()

    #check the format of the file, convert if is tsv or csv, ,and return the json.
    def check_valids_for_tsv_csv_formats(self, path_file, file_name):         
        try:
            with open(path_file, 'r', encoding='utf-8') as f:                
                data = json.load(f) 
                return True, path_file
        except ValueError as e:
            pass               
        
        try:
            with open(path_file, encoding='utf-8') as file:
                rd = csv.reader(file, delimiter=self.DELIMITER_TSV, quotechar='"')
                self.delimiter =self.DELIMITER_TSV
                path = self.parse_tsv_csv_2_json(path_file, file_name)
                self.pyevall_report.insert_file_warning(file_name, PyEvALLReport.FORMAT_TSV_FORMAT_IDENTIFIED_WARNING, False)
                self.logger.debug("Warning %s in file %s in line %s", PyEvALLReport.FORMAT_TSV_FORMAT_IDENTIFIED_WARNING, file_name) 
                return True, path
        except Exception as e:
            pass
        
        try:
            with open(path_file, encoding='utf-8') as file:
                rd = csv.reader(file, delimiter=self.DELIMITER_CSV, quotechar='"')
                self.delimiter =self.DELIMITER_CSV
                path = path = self.parse_tsv_csv_2_json(path_file, file_name)
                self.pyevall_report.insert_file_warning(file_name, PyEvALLReport.FORMAT_CSV_FORMAT_IDENTIFIED_WARNING, False)
                self.logger.debug("Warning %s in file %s in line %s", PyEvALLReport.FORMAT_CSV_FORMAT_IDENTIFIED_WARNING, file_name) 
                return True, path
        except Exception as e:
            pass          
                               
        return False, None

    
           
    def check_file_exist(self, path_file, file_name):
        """
        Check if a file exists and is not empty.
    
        Parameters:
            - path_file: Path to the file to check.
            - file_name: Name of the file.
    
        Returns:
            - Boolean indicating whether the file exists and is not empty.
    
        Functionality:
            - Uses the Path class to create a Path object for the file.
            - Checks if the file exists.
            - Checks if the file is not empty.
            - Inserts file error into the PyEvALLReport if file is empty or does not exist.
            - Logs error message if file is empty or does not exist.
            - Returns True if the file exists and is not empty, otherwise False.
        """
        path = p.Path(path_file)
        if path.exists():
            if path.stat().st_size > 0:
                return True
            else:
                self.pyevall_report.insert_file_error(file_name, PyEvALLReport.FORMAT_EMPTY_FILE_ERROR, None, True)
                self.logger.debug("Error %s in file %s: file empty", PyEvALLReport.FORMAT_EMPTY_FILE_ERROR, file_name)
                return False                
        else:
            self.pyevall_report.insert_file_error(file_name, PyEvALLReport.FORMAT_FILE_NOT_EXIST_ERROR, None, True)
            self.logger.debug("Error %s in file %s: file not exist", PyEvALLReport.FORMAT_FILE_NOT_EXIST_ERROR, file_name)
            return False    
    
     
    def check_repeated_ids(self, lst_ids_per_tc, testcase, id):
        """
        Check if an ID is repeated within a test case.
    
        Parameters:
            - lst_ids_per_tc: Dictionary containing lists of IDs per test case.
            - testcase: Test case to check for repeated IDs.
            - id: ID to check for repetition within the test case.
    
        Returns:
            - Boolean indicating whether the ID is repeated within the test case.
    
        Functionality:
            - Checks if the test case exists in the dictionary.
            - Checks if the ID exists in the list of IDs for the test case.
            - If the test case exists, appends the ID to its list of IDs.
            - If the test case does not exist, creates a new entry with an empty list for IDs and appends the ID.
            - Returns True if the ID is repeated within the test case, otherwise False.
        """
        if testcase in lst_ids_per_tc:
            if id in lst_ids_per_tc[testcase]:
                return True
            else:                
                lst_ids_per_tc[testcase].append(id)
        else:
            lst_ids_per_tc[testcase]=[]
            lst_ids_per_tc[testcase].append(id)
        return False 
        
        
    def get_pyevall_comparators(self):
        """
        Retrieve PyEvALL comparators for each test case.
      
        Returns:
            - lst_comparators: List of PyEvALLComparator instances for each test case.
    
        Functionality:
            - Iterates through each test case in the gold standard data.
            - Checks if the test case exists in the predictions data.
            - Checks consistency of JSON data for the test case.
            - Initializes PyEvALLComparator instance for consistent test cases.
            - Appends initialized comparators to the list.
            - Returns the list of comparators for further evaluation.
        """        
        lst_comparators=[]
        for tc in self.gold_df:
            if tc in self.pred_df:
                if self.check_consistency_json_data(tc, self.pred_df[tc], self.gold_df[tc], self.gold_file_name, self.pred_file_name):
                    comp = comparators.PyEvALLComparator(self.pred_df[tc], self.gold_df[tc], tc, self.evaluation_id)
                    lst_comparators.append(comp)
        return lst_comparators         
        
    
    ##########################################
    #                                        #
    #            PARSER JSON FORMAT          #
    #                                        #
    ##########################################     
    def parse_files_json_format(self):
        """
        Parse JSON-formatted prediction and gold standard files.
    
        Functionality:
            - Parses prediction file into a dictionary if it is valid.
            - Checks the format of the prediction file and converts it into DataFrame for each test case.
            - Inserts file error into the PyEvALLReport if prediction file is empty or invalid.
            - Parses gold standard file into a dictionary if it is valid.
            - Checks the format of the gold standard file and converts it into DataFrame for each test case.
            - Inserts file error into the PyEvALLReport if gold standard file is empty or invalid.
            - Sets valid_execution flag to False if any file contains errors.
        """        
        #if predictions contains errors we stop evaluation and inform
        valid, pred_dict = self.parser_json(self.pred_path, self.pred_file_name)  
        if valid: 
            valid, pred_dict= self.check_format_json(self.pred_file_name, pred_dict, False)
            if len(pred_dict)>0:
                for tc in pred_dict:          
                    self.pred_df[tc]= pd.DataFrame.from_dict(pred_dict[tc])  
            else:
                self.pyevall_report.insert_file_error(self.pred_file_name, PyEvALLReport.FORMAT_EMPTY_FILE_ERROR, None, True)
                self.logger.debug("Error %s in file %s: file empty", PyEvALLReport.FORMAT_EMPTY_FILE_ERROR, self.pred_file_name)
                self.valid_execution=False                 
        else:
            self.valid_execution=False                    
       
        #if gold contains errors we stop evaluation and inform
        valid, gold_dict = self.parser_json(self.gold_path, self.gold_file_name)
        if valid:       
            valid, gold_dict= self.check_format_json(self.gold_file_name, gold_dict, True)  
            if valid: 
                for tc in gold_dict:         
                    self.gold_df[tc]= pd.DataFrame.from_dict(gold_dict[tc])   
            else:
                self.valid_execution=False                                          
        else:
            self.valid_execution=False 
          
               
    def parser_json(self, path, file_name):
        """
        Parse a JSON file and validate its format against a schema.
    
        Parameters:
            - path: Path to the JSON file.
            - file_name: Name of the JSON file.
    
        Returns:
            - Tuple containing a boolean indicating if parsing was successful and the parsed data.
    
        Functionality:
            - Attempts to open and load the JSON file.
            - Inserts file error into the PyEvALLReport if JSON format is incorrect.
            - Validates the JSON data against a predefined schema.
            - Inserts file error into the PyEvALLReport if JSON schema is incorrect.
            - Returns a tuple with a boolean indicating success and the parsed data.
        """
        data = None
        try:
            with open(path, 'r', encoding='utf-8') as f:                
                data = json.load(f) 
        except ValueError as e:
            self.pyevall_report.insert_file_error(file_name, PyEvALLReport.FORMAT_INCORRECT_JSON_ERROR, str(e), True)
            self.logger.debug("Error %s in file %s in line %s", PyEvALLReport.FORMAT_INCORRECT_JSON_ERROR, file_name, e)             
            return False, data
        
        try:
            validate(instance=data, schema=PyEvALLUtils.FORMAT_JSON_SCHEMA)
        except jsonschema.exceptions.ValidationError as e:
            self.pyevall_report.insert_file_error(file_name, PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR, str(e), True)
            self.logger.debug("Error %s in file %s in line %s", PyEvALLReport.FORMAT_INCORRECT_SCHEMA_JSON_ERROR, file_name, e)             
            return False, data
        
        return True, data
    
   
    def check_format_json(self, file_name, data, stop_error):
        """
        Check the format of JSON data and detect repeated IDs within test cases.
    
        Parameters:
            - file_name: Name of the JSON file being checked.
            - data: JSON data to be checked for format and repeated IDs.
            - stop_error: Boolean indicating whether to stop on encountering an error.
    
        Returns:
            - Tuple containing a boolean indicating if there are no errors and the cleaned data.
    
        Functionality:
            - Initializes dictionaries to store clean data and track IDs per test case.
            - Iterates through each instance in the data.
            - Checks for repeated IDs within test cases: if the error is found in the predictions the system continues and inform,
                if the error is found in the gold the process stop.
            - Appends instances to clean data if IDs are not repeated.
            - Inserts file line error into the PyEvALLReport if repeated IDs are found.
            - Logs error message if repeated IDs are found.
            - Returns a tuple with a boolean indicating if there are no errors and the cleaned data.
        """
        clean_data=dict()
        lst_ids_per_tc=dict()
        no_errors=True
        for index, inst in enumerate(data,1):
            if not self.check_repeated_ids(lst_ids_per_tc, inst[PyEvALLFormat.TEST_CASE], inst[PyEvALLFormat.ID]):
                if not inst[PyEvALLFormat.TEST_CASE] in clean_data:
                    clean_data[inst[PyEvALLFormat.TEST_CASE]]=[]
                clean_data[inst[PyEvALLFormat.TEST_CASE]].append(inst)
            else:
                self.pyevall_report.insert_file_line_error(file_name, index, PyEvALLReport.FORMAT_IDS_REPEATED_ROW_ERROR, None, stop_error)
                self.logger.debug("Error %s in file %s in line %s", PyEvALLReport.FORMAT_IDS_REPEATED_ROW_ERROR, file_name, index)
                no_errors=False     
        return no_errors, clean_data    
    
    
    def check_consistency_json_data(self, tc, pred_df, gold_df, gold_file_name, pred_file_name):
        #If gold has different data types in value it is an error.          
        lst_g_type=gold_df[PyEvALLFormat.VALUE].apply(type).unique()
        if len(lst_g_type)!=1:
            self.pyevall_report.insert_file_testcase_error(gold_file_name, tc, PyEvALLReport.FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR, None, True)
            self.logger.debug("Error %s in file %s: file empty", PyEvALLReport.FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR, gold_file_name)
            return False 
                             
        #If predictions has different data types in value it is an error.                            
        lst_p_type=pred_df[PyEvALLFormat.VALUE].apply(type).unique()
        if len(lst_p_type)!=1:
            self.pyevall_report.insert_file_testcase_error(pred_file_name, tc, PyEvALLReport.FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR, None, True)
            self.logger.debug("Error %s in file %s: file empty", PyEvALLReport.FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR, pred_file_name)
            return False
        
        #If they are different types in gold and pred is a mistake.
        if lst_g_type[0]!=lst_p_type[0]:
            self.pyevall_report.insert_file_testcase_error(pred_file_name, tc, PyEvALLReport.FORMAT_DIFFERENT_TYPES_IN_VALUE_GOLD_AND_PRED, None, True)
            self.logger.debug("Error %s in file %s: file empty", PyEvALLReport.FORMAT_DIFFERENT_TYPES_IN_VALUE_GOLD_AND_PRED, pred_file_name)
            return False
        
        return True
            
  
    ##########################################
    #                                        #
    #            WRAPPER TSV AND CSV         #
    #                                        #
    ##########################################      
    def parse_tsv_csv_2_json(self, path_input_file, input_file_name): 
        arr = [] 
        file = open(path_input_file, 'r') 
        a = file.readline() 
          
        # The first line consist of headings of the record  
        # so we will store it in an array and move to  
        # next line in input_file. 
        df_data=None
        
        #We read with pandas to preserver the format
        titles = [t.strip() for t in a.split(self.delimiter)] 
        if not self.check_headers(titles):
            titles[0]=PyEvALLFormat.TEST_CASE
            titles[1]=PyEvALLFormat.ID
            titles[2]=PyEvALLFormat.VALUE       
            df_data = pd.read_csv(path_input_file, sep=self.delimiter, index_col=False, header=None,
                          skip_blank_lines=False, names=titles, dtype={
                            PyEvALLFormat.TEST_CASE: 'str',
                            PyEvALLFormat.ID: 'str',
                            PyEvALLFormat.VALUE: 'str'
                        })   
        else:
            df_data = pd.read_csv(path_input_file, sep=self.delimiter, index_col=False,
                          skip_blank_lines=False, dtype={
                            PyEvALLFormat.TEST_CASE: 'str',
                            PyEvALLFormat.ID: 'str',
                            PyEvALLFormat.VALUE: 'str'
                        })                                
        
        #if all the value columns are numbers convert to ranking format, otherwise mataint str
        try:
            df_data[PyEvALLFormat.VALUE] = pd.to_numeric(df_data[PyEvALLFormat.VALUE])
        except ValueError as e:
            df_data[PyEvALLFormat.VALUE]=df_data[PyEvALLFormat.VALUE].astype("str")

        
        output_file_json= self.get_temp_file(input_file_name)   
        with open(output_file_json, 'w', encoding='utf-8') as output_file: 
            output_file.write(df_data.to_json(orient="records"))            
        
        return output_file_json

    
    def get_temp_file(self, file_name):
        return os.path.join(PyEvALLUtils.get_tmp_dir_execution(), file_name)


    def check_headers(self, row):
        if len(row)==3:
            if row[0]==PyEvALLFormat.TEST_CASE and row[1]==PyEvALLFormat.ID and row[2]==PyEvALLFormat.VALUE:        
                return True
        return False       
    