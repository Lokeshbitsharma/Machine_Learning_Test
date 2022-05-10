# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:09:35 2022

@author: lokes
"""

import pandas as pd
from data_preprocessing.base_data_extractor import BaseDataExtractor

class DataExtractor:
    
    def __init__(self, task_path_dictionary, label_dict):
        
        super().__init__(self,task_path_dictionary,label_dict)
        
    def get_input_output_data(self, path, task_id):
        df = pd.read_csv(path)
        return df
        
        