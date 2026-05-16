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
    """
    Parser, validator, and comparator builder for PyEvALL input files.

    The class detects JSON/TSV/CSV inputs, converts wrappers to JSON records,
    validates schema and semantic constraints, filters non-evaluable rows, and
    creates one `PyEvALLComparator` per valid test case.
    """
    DELIMITER_TSV = "\t"
    DELIMITER_CSV = ","
    TEST_CASE = "test_case"
    ID = "id"
    VALUE = "value" 
    TSV_CSV_POSSIBLE_NULL_VALUES = {"", "nan", "none", "null"}
    

    def __init__(self, pyevall_report, pred_file, gold_file, evaluation_id): 
        # Initialize file paths, report entries, and parsed data containers.
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
        self.ignored_gold_ids_by_tc=dict()
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

    def check_valids_for_tsv_csv_formats(self, path_file, file_name):         
        # Detect JSON directly; otherwise try TSV and CSV wrappers.
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
        # Validate that an input path exists and is non-empty.
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
        # Track repeated ids within each test case.
        if testcase in lst_ids_per_tc:
            if id in lst_ids_per_tc[testcase]:
                return True
            else:                
                lst_ids_per_tc[testcase].append(id)
        else:
            lst_ids_per_tc[testcase]=[]
            lst_ids_per_tc[testcase].append(id)
        return False 

    def is_empty_value(self, value):
        # Treat empty assignments as non-evaluable rows.
        return value is None or value == "" or value == [] or value == {}

    def validate_clusters_in_value_array(self, file_name, index, value, stop_error):
        # Validate semantic constraints for list values used in clustering format.
        if not isinstance(value, list):
            return True

        no_errors = True
        clean_clusters = []
        seen = set()
        for cluster_name in value:
            # The JSON schema already guarantees list[str], but we keep this defensive check.
            if not isinstance(cluster_name, str):
                continue

            # Keep cluster names exactly as provided by the user (no strip).
            cluster_name_clean = cluster_name
            if cluster_name_clean == "":
                self.pyevall_report.insert_file_line_error(
                    file_name,
                    index,
                    PyEvALLReport.FORMAT_CLUSTER_EMPTY_TAG_IN_VALUE_ROW_ERROR,
                    None,
                    stop_error,
                )
                self.logger.debug(
                    "Error %s in file %s in line %s",
                    PyEvALLReport.FORMAT_CLUSTER_EMPTY_TAG_IN_VALUE_ROW_ERROR,
                    file_name,
                    index,
                )
                no_errors = False
                continue

            if cluster_name_clean in seen:
                self.pyevall_report.insert_file_line_error(
                    file_name,
                    index,
                    PyEvALLReport.FORMAT_CLUSTER_DUPLICATED_TAG_IN_VALUE_ROW_ERROR,
                    None,
                    stop_error,
                )
                self.logger.debug(
                    "Error %s in file %s in line %s",
                    PyEvALLReport.FORMAT_CLUSTER_DUPLICATED_TAG_IN_VALUE_ROW_ERROR,
                    file_name,
                    index,
                )
                no_errors = False
                continue

            seen.add(cluster_name_clean)
            clean_clusters.append(cluster_name_clean)

        value[:] = clean_clusters
        return no_errors

    def parse_value_from_tsv_csv(self, value):
        # Parse VALUE from TSV/CSV into JSON-compatible Python types:
        # - int for ranking values
        # - list[str] for clustering/multilabel values
        # - dict for LeWiDi values
        # - str for monolabel values
        if pd.isna(value):
            return ""

        if not isinstance(value, str):
            return value

        # Keep the raw value exactly as written by the user (no strip).
        if value.lower() in self.TSV_CSV_POSSIBLE_NULL_VALUES:
            return ""

        if (value.startswith("[") and value.endswith("]")) or (value.startswith("{") and value.endswith("}")):
            try:
                return json.loads(value)
            except (ValueError, json.JSONDecodeError):
                return value

        try:
            return int(value)
        except ValueError:
            return value
        
        
    def get_pyevall_comparators(self):
        # Build one comparator per common and consistent test case.
        lst_comparators=[]
        for tc in self.gold_df:
            if tc in self.pred_df:
                if self.gold_df[tc].empty:
                    continue
                if self.check_consistency_json_data(tc, self.pred_df[tc], self.gold_df[tc], self.gold_file_name, self.pred_file_name):
                    pred_df = self.pred_df[tc]
                    ignored_gold_ids = self.ignored_gold_ids_by_tc.get(tc, set())
                    if len(ignored_gold_ids) > 0 and not pred_df.empty:
                        pred_df = pred_df[~pred_df[PyEvALLFormat.ID].isin(ignored_gold_ids)]
                    comp = comparators.PyEvALLComparator(pred_df, self.gold_df[tc], tc, self.evaluation_id)
                    lst_comparators.append(comp)
        return lst_comparators         
        
    
    ##########################################
    #                                        #
    #            PARSER JSON FORMAT          #
    #                                        #
    ##########################################     
    def parse_files_json_format(self):
        # Parse, validate, clean, and group prediction/gold records by test case.
        # If predictions contain errors we stop evaluation and inform.
        valid, pred_dict = self.parser_json(self.pred_path, self.pred_file_name)  
        if valid: 
            valid, pred_dict= self.check_format_json(self.pred_file_name, pred_dict, False)
            if len(pred_dict)>0:
                for tc in pred_dict:          
                    self.pred_df[tc]= self.records_to_dataframe(pred_dict[tc])  
            else:
                self.pyevall_report.insert_file_error(self.pred_file_name, PyEvALLReport.FORMAT_EMPTY_FILE_ERROR, None, True)
                self.logger.debug("Error %s in file %s: file empty", PyEvALLReport.FORMAT_EMPTY_FILE_ERROR, self.pred_file_name)
                self.valid_execution=False                 
        else:
            self.valid_execution=False                    
       
        # If gold contains errors we stop evaluation and inform.
        valid, gold_dict = self.parser_json(self.gold_path, self.gold_file_name)
        if valid:       
            valid, gold_dict= self.check_format_json(self.gold_file_name, gold_dict, True)  
            if valid: 
                for tc in gold_dict:         
                    self.gold_df[tc]= self.records_to_dataframe(gold_dict[tc])   
            else:
                self.valid_execution=False                                          
        else:
            self.valid_execution=False 
          
               
    def parser_json(self, path, file_name):
        # Load a JSON file and validate it against the PyEvALL schema.
        data = None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, OSError) as e:
            self.pyevall_report.insert_file_error(file_name, PyEvALLReport.FORMAT_FILE_NOT_EXIST_ERROR, str(e), True)
            self.logger.debug("Error %s in file %s in line %s", PyEvALLReport.FORMAT_FILE_NOT_EXIST_ERROR, file_name, e)
            return False, data
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

    def records_to_dataframe(self, records):
        # Build a dataframe while preserving required columns for empty test cases.
        df_data = pd.DataFrame.from_dict(records)
        if df_data.empty:
            return pd.DataFrame(columns=[PyEvALLFormat.TEST_CASE, PyEvALLFormat.ID, PyEvALLFormat.VALUE])
        return df_data
    
   
    def check_format_json(self, file_name, data, stop_error):
        # Clean JSON records, warning on empty values and rejecting repeated ids.
        clean_data=dict()
        lst_ids_per_tc=dict()
        no_errors=True
        for index, inst in enumerate(data,1):
            clusters_valid = self.validate_clusters_in_value_array(
                file_name, index, inst[PyEvALLFormat.VALUE], stop_error
            )
            if self.is_empty_value(inst[PyEvALLFormat.VALUE]):
                if not inst[PyEvALLFormat.TEST_CASE] in clean_data:
                    clean_data[inst[PyEvALLFormat.TEST_CASE]]=[]
                if stop_error:
                    if not inst[PyEvALLFormat.TEST_CASE] in self.ignored_gold_ids_by_tc:
                        self.ignored_gold_ids_by_tc[inst[PyEvALLFormat.TEST_CASE]]=set()
                    self.ignored_gold_ids_by_tc[inst[PyEvALLFormat.TEST_CASE]].add(inst[PyEvALLFormat.ID])
                self.pyevall_report.insert_file_line_warning(file_name, index, PyEvALLReport.FORMAT_EMPTY_VALUE_ROW_WARNING, None, False)
                self.logger.debug("Warning %s in file %s in line %s", PyEvALLReport.FORMAT_EMPTY_VALUE_ROW_WARNING, file_name, index)
                if not clusters_valid:
                    no_errors=False
                continue
            if not self.check_repeated_ids(lst_ids_per_tc, inst[PyEvALLFormat.TEST_CASE], inst[PyEvALLFormat.ID]):
                if not inst[PyEvALLFormat.TEST_CASE] in clean_data:
                    clean_data[inst[PyEvALLFormat.TEST_CASE]]=[]
                clean_data[inst[PyEvALLFormat.TEST_CASE]].append(inst)
            else:
                self.pyevall_report.insert_file_line_error(file_name, index, PyEvALLReport.FORMAT_IDS_REPEATED_ROW_ERROR, None, stop_error)
                self.logger.debug("Error %s in file %s in line %s", PyEvALLReport.FORMAT_IDS_REPEATED_ROW_ERROR, file_name, index)
                no_errors=False
            if not clusters_valid:
                no_errors=False
        return no_errors, clean_data    
    
    
    def check_consistency_json_data(self, tc, pred_df, gold_df, gold_file_name, pred_file_name):
        # Ensure gold and prediction values use compatible types for a test case.
        lst_g_type=gold_df[PyEvALLFormat.VALUE].apply(type).unique()
        if len(lst_g_type)!=1:
            self.pyevall_report.insert_file_testcase_error(gold_file_name, tc, PyEvALLReport.FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR, None, True)
            self.logger.debug("Error %s in file %s: file empty", PyEvALLReport.FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR, gold_file_name)
            return False 

        if pred_df.empty:
            return True
                             
        # If predictions has different data types in value it is an error.
        lst_p_type=pred_df[PyEvALLFormat.VALUE].apply(type).unique()
        if len(lst_p_type)!=1:
            self.pyevall_report.insert_file_testcase_error(pred_file_name, tc, PyEvALLReport.FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR, None, True)
            self.logger.debug("Error %s in file %s: file empty", PyEvALLReport.FORMAT_DIFFERENTE_TYPES_IN_VALUE_FIELD_ERROR, pred_file_name)
            return False
        
        # If they are different types in gold and pred is a mistake.
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
        # Read a TSV/CSV wrapper and write an equivalent temporary JSON file.
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
        
        # Convert VALUE to native JSON types so clustering arrays are preserved.
        df_data[PyEvALLFormat.VALUE] = df_data[PyEvALLFormat.VALUE].apply(self.parse_value_from_tsv_csv)

        
        output_file_json= self.get_temp_file(input_file_name)   
        with open(output_file_json, 'w', encoding='utf-8') as output_file: 
            output_file.write(df_data.to_json(orient="records"))            
        
        return output_file_json

    
    def get_temp_file(self, file_name):
        # Return the temporary path used for wrapper conversion.
        return os.path.join(PyEvALLUtils.get_tmp_dir_execution(), file_name)


    def check_headers(self, row):
        # Detect whether the first TSV/CSV row contains PyEvALL headers.
        if len(row)==3:
            if row[0]==PyEvALLFormat.TEST_CASE and row[1]==PyEvALLFormat.ID and row[2]==PyEvALLFormat.VALUE:        
                return True
        return False       
    
