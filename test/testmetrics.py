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
import pandas as pd
import math
from pyevall.evaluation import PyEvALLEvaluation
from pyevall.metrics.metricfactory import MetricFactory
from pyevall.reports.reports import PyEvALLReport
from pyevall.utils.utils import PyEvALLUtils

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKCYAN = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def test_results_df(df_pyevall_imp, df_double_imp, metric_conv):        
    passed=True
    if len(df_pyevall_imp)==0:
        passed=False
    for row in df_pyevall_imp.iterrows():
        name_gold=metric_conv[row[1].metric]
        
        if row[1].score==None:
            print(FAIL + "\tError" + ENDC+" in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (name_gold, float(df_double_imp[df_double_imp["metric"] == name_gold]["score"]), row[1].score))
            passed=False        
        else: 
            score_pyevall=float("{:.4f}".format(row[1].score))
            score_gold=float("{:.4f}".format(float(df_double_imp[df_double_imp["metric"] == name_gold]["score"])))
    
            if not score_gold==score_pyevall:
                if not (math.isnan(score_gold) and math.isnan(score_pyevall)):
                    print(FAIL + "\tError" + ENDC+" in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (row[1].metric, score_gold, score_pyevall))
                    passed=False
                else:
                    print(OKGREEN + "\tOK" + ENDC + " in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (row[1].metric, score_gold, score_pyevall))
            else:
                print(OKGREEN + "\tOK" + ENDC + " in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (row[1].metric, score_gold, score_pyevall))
    return passed


def parse_results_double_imp(path):
    df = pd.read_csv(path, sep="\t", dtype = str, encoding='utf-8',names=["metric", "score"])
    return df


#################################################################
#
#    Test classification metrics
#
#################################################################
metric_conversion_2={ 'Acc': 'ACC', 
                   'Precision': 'precision', 
                   'Recall':'recall',
                   'FMeasure':'fmeasure',
                   'SP':'SysPre',
                   'Kappa':'Kappa',
                   'ICM':'ICM'
                    }


metric_conversion_3={ 
                   'Pr_TRUE': 'Precision_TRUE', 
                   'Re_TRUE':'Recall_TRUE',
                   'F1_TRUE':'Fmeasure_TRUE'
                    }


def test_2_classification_metrics():
    file_double_imp="resources/metric/double_imp/classification/Double_imp_2.txt"
    folder_pred="resources/metric/test/classification/predictions/1/"
    folder_gold="resources/metric/test/classification/gold/1/"
    file_name_sys= "SYS"
    file_name_gold= "GOLD"
    params={PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE }
    df_double_imp= pd.read_csv(file_double_imp, sep='\t')
    
    
    for i in range(1,26):
        print("Executing test number %s: ", i)
        pred= folder_pred + file_name_sys + str(i) + ".txt"
        gold= folder_gold + file_name_gold + str(i) + ".txt"
        
        df_pyevall_report= evaluate_pyevall_classification_2(pred, gold, **params)
        row_double_imp= (i-1)*2+1
        
        for index, row in df_pyevall_report.df_average.iterrows():
            columns = list(df_pyevall_report.df_average)
            passed = True
            for c in columns:
                if c == "files":
                    continue
                result_pyevall= row[c]
                if c in metric_conversion_2:
                    result_double_imp = df_double_imp.iloc[row_double_imp, df_double_imp.columns.get_loc(metric_conversion_2[c])]
                    passed_metric= test_results_df_2(c, result_double_imp, result_pyevall)
                    passed = passed and passed_metric
        
        if not passed:
            print("\tTest 1 number %s " % (i), FAIL + "FAILED" + ENDC)
        else:
            print("\tTest 1 number %s " % (i), OKGREEN + "PASSED" + ENDC)            
        
        
        if df_pyevall_report.df_test_case_classes is not None:
            for index, row in df_pyevall_report.df_test_case_classes.iterrows():
                columns = list(df_pyevall_report.df_test_case_classes)
                passed = True
                for c in columns:
                    if c == "files":
                        continue
                    result_pyevall= row[c]
                    if c in metric_conversion_3:
                        result_double_imp = df_double_imp.iloc[row_double_imp, df_double_imp.columns.get_loc(metric_conversion_3[c])]
                        passed_metric= test_results_df_2(c, result_double_imp, result_pyevall)
                        passed = passed and passed_metric            
            
            if not passed:
                print("\tTest 2 number %s " % (i), FAIL + "FAILED" + ENDC)
            else:
                print("\tTest 2 number %s " % (i), OKGREEN + "PASSED" + ENDC)
        else:
            print("\tTest 2 number %s " % (i), FAIL + "FAILED" + ENDC)
        

def evaluate_pyevall_classification_2(pred, gold, **params):
    test = PyEvALLEvaluation()
    m = [MetricFactory.Accuracy.value, MetricFactory.Precision.value, MetricFactory.Recall.value, MetricFactory.FMeasure.value, "SystemPrecision", "Kappa", MetricFactory.ICM.value]
    params[PyEvALLUtils.PARAM_REPORT]= PyEvALLUtils.PARAM_OPTION_REPORT_DATAFRAME   
    return test.evaluate(pred, gold, m, **params)


def test_results_df_2(metric, score_double_imp, score_pyevall):        
    passed=True
    if score_double_imp=="UNK" and score_pyevall=="-":
        print(OKGREEN + "\t\tOK" + ENDC + " in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (metric, score_double_imp, score_pyevall))
        return passed
    if score_double_imp=="UNK" and score_pyevall==None:
        print(OKGREEN + "\t\tOK" + ENDC + " in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (metric, score_double_imp, score_pyevall))
        return passed
    elif score_double_imp=="UNK":
        print(FAIL + "\tError" + ENDC+"  in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (metric, score_double_imp, score_pyevall))
        return False
    elif score_pyevall=="-":
        print(FAIL + "\tError" + ENDC+"  in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (metric, score_double_imp, score_pyevall))
        return False
    elif score_pyevall==None:
        print(FAIL + "\tError" + ENDC+" in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (metric, score_double_imp, score_pyevall))
        return False        
    else: 
        score_pyevall= float("{:.3f}".format(float(score_pyevall)))  
        score_double_imp = float("{:.3f}".format(float(score_double_imp)))  
        if not score_double_imp==score_pyevall:
            if not (math.isnan(score_double_imp) and math.isnan(score_pyevall)):
                print(FAIL + "\tError" + ENDC+"  in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (metric, score_double_imp, score_pyevall))
                passed=False
            else:
                print(OKGREEN + "\t\tOK" + ENDC + " in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (metric, score_double_imp, score_pyevall))
        else:
            print(OKGREEN + "\t\tOK" + ENDC + " in metric %s:  Value-gold: %s --- Value-pyevall: %s" % (metric, score_double_imp, score_pyevall))
    return passed      


#################################################################
#
#    Test Ranking metrics
#
#################################################################
metric_conversion_ranking={ 'PrecisionAtK': 'precisionatk', 
                   'RPrecision': 'rprecision', 
                   'MRR':'mainreciprocalrank',
                   'MAP':'map',
                   'DCG':'dcg',
                   'nDCG':'ndcg',
                   'ERR':'err',
                   'RBP':'rbp'
                    }
#RBP uses param_p=0.5

def test_1_ranking_metrics():
    folder_double_imp="resources/metric/double_imp/ranking/"
    folder_pred="resources/metric/test/ranking/predictions/"
    folder_gold="resources/metric/test/ranking/gold/"
    file_name_sys= "SYS"
    file_name_gold= "GOLD"
    params={PyEvALLUtils.PARAM_LOG_LEVEL: PyEvALLUtils.PARAM_OPTION_LOG_LEVEL_NONE }
    for i in range(1,16):
        if i==4:
            continue
        print("Executing test number %s: ", i)
        file_double_imp= folder_double_imp + file_name_sys + str(i) + ".txt"
        pred= folder_pred + file_name_sys + str(i) + ".txt"
        gold= folder_gold + file_name_gold + str(i) + ".txt"
        df_double_imp = parse_results_double_imp(file_double_imp)
        df_pyevall_imp = evaluate_pyevall_ranking(pred, gold, **params)
        if not test_results_df(df_pyevall_imp, df_double_imp, metric_conversion_ranking):
            print("\tTest number %s " % (i), FAIL + "FAILED" + ENDC)
        else:
            print("\tTest number %s " % (i), OKGREEN + "PASSED" + ENDC)

   
def evaluate_pyevall_ranking(pred, gold, **params):
    prueba = PyEvALLEvaluation()
    m = [MetricFactory.PrecisionAtK.value, MetricFactory.RPrecision.value, MetricFactory.MRR.value, MetricFactory.MAP.value, \
         MetricFactory.DCG.value, MetricFactory.nDCG.value, MetricFactory.ERR.value, MetricFactory.RBP.value]
    report_object = prueba.evaluate(pred, gold, m, **params)
    report= report_object.report
    
    metrics = report[PyEvALLReport.METRIC_TAG]
    metric_names = []
    score = []
    for metric in metrics:
        metric_names.append(metric)
        if PyEvALLReport.AVERAGE_PER_TC_TAG in metrics[metric][PyEvALLReport.RESULTS_TAG]:
            score.append(metrics[metric][PyEvALLReport.RESULTS_TAG][PyEvALLReport.AVERAGE_PER_TC_TAG])
        else:
            score.append(None)
    d = {'metric':metric_names, 'score':score}
    df = pd.DataFrame(d)
    return df





if __name__ == '__main__':    
    test_2_classification_metrics()
    test_1_ranking_metrics()

    
    
    
    
    
    
    
    
    
    
    