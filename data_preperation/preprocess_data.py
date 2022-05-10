# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 08:23:57 2021


"""

import numpy as np
import pandas as pd

from pickle import dump, load
import configparser
import ast

class LoadTransformData:
    
    def __init__(self, config_path):    
        '''
        Load Preprocessed objects and perform transformation in data.
        Parameters
        ----------
        config_path : str
            configuration file path.

        Returns
        -------
        None.

        '''
        
        self.parser = configparser.ConfigParser()
        self.parser.read(config_path)
        self.generate_feature_from_config()
        
        self.__load_categorical_transform()
        self.__load_standard_scaler()
        
    
    def transform_input_data(self, X):  
        '''
        Transform input data
        Parameters
        ----------
        X : Pandas Dataframe
            Input data.
        Returns
        -------
        X : Pandas Dataframe
            Transformed data.

        '''
        X.loc[:,self.STR_COLUMNS] = self.categorical_transform.transform(X.loc[:,self.STR_COLUMNS])
        std_column_list = self.INT_COLUMNS + self.STR_COLUMNS
        X.loc[:,std_column_list] = self.std_scaler_transform.transform(X.loc[:,std_column_list])
        return X
    
    def inverse_transform_data(self,x_transformed):
        # df = pd.DataFrame(columns = self.FEATURE_COLUMNS )
        df = x_transformed.copy()        
        index = [i for i in x_transformed.index]
        
        std_column_list = self.INT_COLUMNS + self.STR_COLUMNS
        df.loc[index, std_column_list] = self.std_scaler_transform.named_transformers_['std_scaler'].inverse_transform(df[std_column_list])
        
        df[df.columns] = df.values.astype(int)
        
        df.loc[index, self.STR_COLUMNS] = self.categorical_transform.named_transformers_['categorical_data_processing'].inverse_transform(df[self.STR_COLUMNS].values)
       
        return df
        
    def __load_categorical_transform(self):
        '''
        Load Catagorical transformation object.
        Returns
        -------
        None.

        '''
        cathegorical_transform_path = ast.literal_eval(self.parser.get("preprocess_dump_path","categorical_transform"))
        with open(cathegorical_transform_path, "rb") as input_file:
            self.categorical_transform = load(input_file)
            
            
    def __load_standard_scaler(self):
        '''
        Load Standard Scaler Object

        Returns
        -------
        None.

        '''
        std_scaler_transform_path = ast.literal_eval(self.parser.get("preprocess_dump_path","std_scaler_transform"))
        with open(std_scaler_transform_path, "rb") as input_file:
            self.std_scaler_transform = load(input_file)
            
    def generate_feature_from_config(self):
        '''
        Generate Feature from Config file

        Returns
        -------
        None.

        '''
        self.LABEL_COLUMN = ast.literal_eval(self.parser.get("feature_specific","LABEL_COLUMN"))    
        self.BOOL_COLUMNS = ast.literal_eval(self.parser.get("feature_specific","BOOL_COLUMNS"))
        self.INT_COLUMNS = ast.literal_eval(self.parser.get("feature_specific","INT_COLUMNS"))
        self.STR_COLUMNS = ast.literal_eval(self.parser.get("feature_specific","STR_COLUMNS"))
        self.STR_NUNIQUESS = ast.literal_eval(self.parser.get("feature_specific","STR_NUNIQUESS"))
        self.FLOAT_COLUMNS = ast.literal_eval(self.parser.get("feature_specific","FLOAT_COLUMNS"))

        self.FEATURE_COLUMNS = (self.INT_COLUMNS + self.BOOL_COLUMNS + self.STR_COLUMNS + self.FLOAT_COLUMNS)
        self.ALL_COLUMNS = self.FEATURE_COLUMNS + [self.LABEL_COLUMN]