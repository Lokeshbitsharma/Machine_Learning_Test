# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:14:53 2022

@author: lokes
"""

import os
import numpy as np
import pandas as pd
from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

class FeatureScalerType(Enum):
    standard_scaler = 1,
    minmax_scaler = 2
    
class CustomPreprocessing:
    
    def __init__(self,df, input_labels, output_labels, index_label = None):
        self.df = df
        self.index_label = index_label
        self.df[self.index_label] = pd.to_datetime(self.df[self.index_label])
        
        if index_label is not None:
            self.df = self.set_indexlabel_in_df()
            
        if any(True for output in output_labels if output in input_labels):
            input_output_label = input_labels
        else:
            input_output_label = input_labels + output_labels
            
        self.df = df[input_output_label]
        self.input_labels = input_labels
        self.output_labels = output_labels
        
    def set_indexlabel_in_df(self):
        if self.index_label not in self.df.columns:
            raise ValueError('Feature with name: %s, does not exists in dataframe.'%self.index_label)
            
        if self.df.index.name != self.index_label:
            self.df.index = self.df[self.index_label]
            self.df = self.df.drop(self.index_label,axis=1)
            
        return self.df
    
    def feature_scaling(self, task_id, scaling_type_feature_dict, 
                        scaler_storage_path, retrieve_scaler=False):
        
        for scaling_type, feature in scaling_type_feature_dict.items():
            scaler_name = scaling_type.split('_')[0]
            scaler = None
            if retrieve_scaler:
                self._load_scale_feature(task_id, feature, scaler_storage_path)
                
            else:
                if scaler_name in FeatureScalerType.standard_scaler.name:
                    scaler = StandardScaler()
                elif scaler_name in FeatureScalerType().minmax_scaler.name:
                    scaler = MinMaxScaler()
                    
                self.__scale_feature(scaler, task_id, feature, scaler_storage_path)
                
        return self.df
    
    def __save_scaling_object(self, task_id, features, path, scaling_obj):
        path = self.__get_path(task_id, features, path)
        joblib.dump(scaling_obj, path)
        
    def __get_path(self, task_id, features, path):
        name = str()
        for i in features:
            name = name + i + '_'
        path = path + '/' + name + str(task_id)
        path = path + '.pkl'
        return path
        
    def __scale_feature(self, scaler, task_id, features, scaler_storage_path):
        self.df[features] = scaler.fit_transform(self.df[features])
        self.__save_scaling_object(task_id, features, scaler_storage_path, scaler)
        
    def __load_scale_feature(self, task_id, features, scaler_storage_path):
        path = self.__get_path(task_id, features, scaler_storage_path)
        scaler = joblib.load(path)
        self.df[features] = scaler.transform(self.df[features])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        
    