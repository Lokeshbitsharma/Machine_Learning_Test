# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 07:25:43 2022

@author: lokes
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import activations 
from timeseries_window import WindowGenerator

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneOut,KFold

from pickle import dump



def get_time_series_window_data(cluster_df_dict_basedon_hour_gap, min_max_scaler, input_width, feature_column, time_column):
    '''
    This method creat dataset based on input_width and creates data list.

    Parameters
    ----------
    cluster_df_dict_basedon_hour_gap : dict
        DESCRIPTION.
    min_max_scaler : TYPE
        DESCRIPTION.
    input_width : int
        DESCRIPTION.
   

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    window_generated_data_list = list()
    window_timestamp_list = list()    
    
    
    for i in list(cluster_df_dict_basedon_hour_gap.keys()):
        if  cluster_df_dict_basedon_hour_gap[i].shape[0]>=input_width :
            gen = WindowGenerator(input_width)

            scaled_cluster_data = cluster_df_dict_basedon_hour_gap[i][feature_column].copy()
            scaled_cluster_data[feature_column] = min_max_scaler.transform(cluster_df_dict_basedon_hour_gap[i][feature_column])        
            scaled_cluster_data.index = cluster_df_dict_basedon_hour_gap[i].index
            
            window_generated_data_list.extend(gen.get_train_data(scaled_cluster_data))
            
            cluster_timestamp = cluster_df_dict_basedon_hour_gap[i][time_column].copy()
            cluster_timestamp.index = cluster_df_dict_basedon_hour_gap[i].index
            window_timestamp_list.extend(gen.get_train_data(cluster_timestamp))
            
    return np.array(window_generated_data_list), np.array(window_timestamp_list)



def get_cluster_df_basedon_gap(data_v1, index_hour_greater_than_1):
    cluster_df_dict_basedon_hour_gap = dict()
    for i, index_ in enumerate(index_hour_greater_than_1 ):
        if i+1 < len(index_hour_greater_than_1):
            cluster_name = 'cluster_' + str(i)
            if i == 0:
                
                cluster_df_dict_basedon_hour_gap[cluster_name] = data_v1.loc[:index_hour_greater_than_1[i]]
                cluster_df_dict_basedon_hour_gap[cluster_name] = cluster_df_dict_basedon_hour_gap[cluster_name].drop([index_hour_greater_than_1[i]],0)
            else:
                cluster_df_dict_basedon_hour_gap[cluster_name] = data_v1.loc[index_hour_greater_than_1[i]:index_hour_greater_than_1[i+1]] 
                cluster_df_dict_basedon_hour_gap[cluster_name] = cluster_df_dict_basedon_hour_gap[cluster_name].drop([index_hour_greater_than_1[i+1]],0)
                
    for k,v in cluster_df_dict_basedon_hour_gap.items():
        print('Clusters: {}, Shape: {}'.format(k,v.shape))
        
    return cluster_df_dict_basedon_hour_gap


def get_train_validate_dataset(data, batch_size=128, n_splits = 5):
    '''
    
    Yields
    ------
    tf.data.Dataset
        Train and Test data based on kFold n_splits.

    '''
    
    def gen():
        kf = KFold(n_splits= n_splits)    
        for train_index, test_index in kf.split(np.arange(0,data.shape[0],1,dtype= np.int64)):            
            
            train, test = tf.convert_to_tensor(data[train_index], dtype=tf.float32) , tf.convert_to_tensor(data[test_index], dtype=tf.float32) 
                       
            train_dataset = tf.data.Dataset.from_tensor_slices(train)
            train_dataset = train_dataset.batch(batch_size)
            
            test_dataset = tf.data.Dataset.from_tensor_slices(test)
            test_dataset = test_dataset.batch(batch_size)
            
            yield train_dataset, test_dataset
        
    return gen()

def split_train_test_data(data, split_ratio = 0.8):
    '''
    
    Parameters
    ----------
    data : pandas data frame
        DESCRIPTION.
    split_ratio : float, optional
        DESCRIPTION. The default is 0.8.

    Returns
    -------
    train_data : pandas dataframe
        DESCRIPTION.
    test_data : pandas dataframe
        DESCRIPTION.

    '''
    split_index = int(len(data) * 0.8)
    train_data = data[0:split_index]
    test_data = data[split_index:]
    
    return train_data, test_data
