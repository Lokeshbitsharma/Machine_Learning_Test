# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 17:14:00 2021


"""
import os
import shutil
import numpy as np
import pandas as pd

from pickle import dump, load
import configparser
import ast
import sklearn
from tensorflow.io import gfile
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from data_preparation.datasampling import DataSampling
from data_preparation.feature_meta_data import FeatureMetaData

def validate_path(result_path):
    path_ = None
    for path in result_path.split('/'):
      if path_ is None:
          path_ = path
      else:
          path_ = path_ + '/' + path
      if not gfile.exists(path_):
          gfile.mkdir(path_)


class GermanDataPreperation:
    
    def __init__(self, data_path, preprocess_dump_path, config_path):
        '''
        
        Parameters
        ----------
        data_path : str
            Data Path.
        preprocess_dump_path : str
            Path where preprocessing pkl object needs to be dumpes.
        config_path : str
            Configuration file path.

        Returns
        -------
        None.

        '''
        
        self.df = pd.read_csv(data_path)
        self.preprocess_dump_path = preprocess_dump_path  
        self.feature_metadata = FeatureMetaData()        
        self.preprocess_data()
            
        
    def get_stratified_train_test_data(self, test_size = 0.3, random_state = 0, 
                                       shuffle = True): 
        '''
        Get stratified Train-Test data.
        Parameters
        ----------
        test_size : float, optional
            Test size. The default is 0.3.
        random_state : int, optional
            Random State. The default is 0.
        shuffle : bool, optional
            Shuffle data. The default is True.

        Returns
        -------
        X_train : Pandas Dataframe
            Input Train data.
        X_test : Pandas Dataframe
            input Test data.
        y_train : Pandas Dataframe
            Output - Train data.
        y_test : Pandas Dataframe
            Output Test Data.

        '''
        
        data_sampling = DataSampling(self.df, self.feature_metadata .FEATURE_COLUMNS, 
                                     self.feature_metadata.LABEL_COLUMN)
        
        X_train, X_test, y_train, y_test = data_sampling.stratified_sampling(test_size = test_size,
                                                                             random_state = random_state,
                                                                             shuffle = shuffle)
        
        return X_train, X_test, y_train, y_test
    
    def get_stratified_kfold_data(self, n_splits = 5, shuffle = True):
        '''
        Get Stratified KFold data.
        Parameters
        ----------
        n_splits : int, optional
            Number of splits. The default is 5.
        shuffle : bool, optional
            Shuffle data. The default is True.

        Returns
        -------
        X : Pandas Dataframe
            Input data.
        y : Pandas Dataframe
            output data.
        kfold : object
            kfold instance.

        '''
        data_sampling = DataSampling(self.df, self.feature_metadata.FEATURE_COLUMNS, 
                                     self.feature_metadata.LABEL_COLUMN)        
        X, y, kfold = data_sampling.stratified_kfold(n_splits,shuffle)        
        return X, y ,kfold
        
    def get_initialbias(self):
        '''
        Returns
        -------
        FLOAT
            Bias to be initialized in model in case of imbalance data set.

        '''
        neg, pos = np.bincount(self.df[self.feature_metadata.LABEL_COLUMN])
        return np.log([pos/neg])
    
    
    def preprocess_data(self):     
        '''
        Preprocess Input Data

        Returns
        -------
        None.

        '''
        self.replace_column_value('GoodCustomer', {-1:0})         
        self.process_categorical_column(self.feature_metadata.STR_COLUMNS)         
        std_column_list = self.feature_metadata.INT_COLUMNS + self.feature_metadata.STR_COLUMNS
        self.transform_standard_scaler(std_column_list)       
   
        
    def replace_column_value(self, column_name, value_dict):
        '''
        Replace value in dataframe for a column(column_name) with value provided in dictionary.

        Parameters
        ----------
        column_name : str
            Feature whose value needs to be changed.
        value_dict : dict
            key: Actual value to be replaced; value: value to replace actual value.

        Raises
        ------
        ValueError
            In case feature is not avaialble in dataframe.

        Returns
        -------
        None.

        '''
        if column_name in self.df.columns:
            self.df[column_name] = self.df[column_name].replace(value_dict)
        else:
            raise ValueError(" Feature {} doesn't exist in the data frame.".format(column_name))
                
                    
                    
    def process_categorical_column(self, column_name_list, dtype =  np.int64 ):     
        '''
        Preprocess Categorical column and dump pkl object in dump path.
        Parameters
        ----------
        column_name_list : list
            List of categorical columns.
        dtype : Data Type, optional
            DESCRIPTION. The default is np.int64.

        Returns
        -------
        None.

        '''
        validate_path(self.preprocess_dump_path)
        column_transform = ColumnTransformer([('categorical_data_processing',OrdinalEncoder(dtype = dtype), column_name_list)],
                                             remainder='passthrough')
        self.df[column_name_list] = column_transform.fit_transform(self.df[column_name_list])
        dump(column_transform, open(self.preprocess_dump_path + '/' + 'categorical_data_processing' +'.pkl', 'wb'))
        
    
                
    def transform_standard_scaler(self,column_list):
        '''
        Standardize data

        Parameters
        ----------
        column_list : list
            Feature list to be standardized.

        Returns
        -------
        None.

        '''
        validate_path(self.preprocess_dump_path)
        column_transform = ColumnTransformer([('std_scaler', StandardScaler(), column_list)], remainder='passthrough')
        self.df[column_list] = column_transform.fit_transform(self.df[column_list])
        dump(column_transform, open(self.preprocess_dump_path + '/' + 'std_scaler' +'.pkl', 'wb'))
        
    
        
    