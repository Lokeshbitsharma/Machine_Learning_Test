# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:54:40 2022

@author: lokes
"""

import pandas as pd
from data_preprocessing.base_data_extractor import BaseDataExtractor
from data_preprocessing.custom_preprocessing import CustomPreprocessing

class TimeSeriesDataExtractor(BaseDataExtractor):
    
    def __init__(self, task_path_dict, label_dict,
                 config_params, retrieve_scaler = False):
        
        self.scaler_feature_dict = config_params.get_scalertype_feature_dict()
        self.scaler_storage_path = config_params.get_scaler_storage_path()
        self.retrieve_scaler = retrieve_scaler
        
        super().__init__(task_path_dict,label_dict)
        
    def get_input_output_data(self, path, task_id):
        df = pd.read_csv(path)
        
        path_split= path.splt('/')
        for i in range(len(path_split)-1,1,-1):
            if'.' not in path_split[i]:
                task_id = path_split[i]
                break
        
        preprocessing = CustomPreprocessing(df,self.input_label, self.output_label,self.index_label)
        
        df = preprocessing.feature_scaling(task_id, self.scaler_feature_dict,
                                           self.scaler_storage_path, self.retrieve_scaler)
        return df