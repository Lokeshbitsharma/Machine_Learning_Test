# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 07:21:19 2021

"""

import configparser
import ast
import os
from data_preparation.feature_meta_data import FeatureMetaData
import pandas as pd
import numpy as np
from constants import Constants

class DataPropertyConfig:
    
    def __init__(self):        
        self.feature_meta_data = FeatureMetaData()
        self.data_prop_config_path = Constants.data_property_config_path
        
    def generate_dataproperty_config(self, df):
        
        '''        
        Parameters
        ----------
        X : pandas dataframe
            Based on this config file for data properties is generated 

        Returns
        -------
        None.

        '''
        if type(df) !=type(pd.DataFrame()):
            raise ValueError('Invalid input data type. Expected data type is pandas dataframe.')            
        
             
        config = configparser.ConfigParser()
        
        
        for feature in self.feature_meta_data.INT_COLUMNS:
            config.add_section(feature)
            config.set(feature,'min',str(df[feature].min()))
            config.set(feature,'max',str(df[feature].max()))
            config.set(feature,'type',str(df[feature].dtype))
            
            
        for feature in self.feature_meta_data.BOOL_COLUMNS:
            config.add_section(feature)
            config.set(feature,'value',str(list(set(df[feature]))))
            config.set(feature,'type',str(df[feature].dtype))
            
        for feature in self.feature_meta_data.STR_COLUMNS:
            config.add_section(feature)
            config.set(feature,'value',str(list(set(df[feature]))))
            config.set(feature,'type','str')
           
        with open(self.data_prop_config_path, 'w') as configfile:
            config.write(configfile)
            
    def fetch_dataproperty_from_config(self):
        '''
        
        Returns
        -------
        fproperty : Python Dictionary
            Gives dictionary to build schema of the dataset

        '''  
        fproperty = dict()
        parser = configparser.ConfigParser()
        parser.read(self.data_prop_config_path)
        
        for feature in self.feature_meta_data.INT_COLUMNS:
            int_feature=dict()
            int_feature['min'] = ast.literal_eval(parser.get(feature,"min"))
            int_feature['max'] = ast.literal_eval(parser.get(feature,"max"))
            int_feature['type'] = parser.get(feature,"type")
            fproperty[feature]=int_feature
            
        for feature in self.feature_meta_data.BOOL_COLUMNS:
            bool_feature = dict()
            bool_feature['type'] = parser.get(feature,"type")
            bool_feature['value'] = ast.literal_eval(parser.get(feature,"value"))
            fproperty[feature] = bool_feature
            
        for feature in self.feature_meta_data.STR_COLUMNS:
            str_feature=dict()
            str_feature['type'] = parser.get(feature,"type")
            str_feature['value'] = ast.literal_eval(parser.get(feature,"value"))
            
            fproperty[feature] = str_feature            
            
        return fproperty
