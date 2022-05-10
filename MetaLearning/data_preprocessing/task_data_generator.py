# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 20:21:00 2022

@author: lokes
"""

import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing.time_series_window_generator import WindowGenerator


class TaskDataGenerator:
    
    def __init__(self, K= 10, n_test_points = 3, label_dict = None, isTimeSeriesData = False):
        if label_dict is None:
            raise ValueError('Label dictionary can not be None.')
            
        self.K = K
        self.n_test_points = n_test_points
        self.input_labels = label_dict['input_label']
        self.output_labels = label_dict['output_label']
        self.isTimeSeriesData = isTimeSeriesData
        
    def train_test_generator(self, df, timeseries_params = None):
        if self.isTimeSeriesData:
            if timeseries_params is None:
                raise ValueError("Time Series params can not be null for time series data.")
            return self.__timeseries_data_generator(df, timeseries_params)
        else:
            return self.__sample_train_test_generator(df)
        
    def __sample_train_test_generator(self,df):
        n_test_finish_index = self.K + self.n_test_points
        
        df = df.sample(frac=1)
        X,y = df[self.input_labels], df[self.output_labels]
        
        X_train = X[:self.K]
        y_train = y[:self.K]
        
        if n_test_finish_index > df.shape[0]:
            X_test = X[self.K:]
            y_test = y[self.K:]
        else:
            X_test = X[self.K : n_test_finish_index]
            y_test = y[self.K: n_test_finish_index]
            
        return X_train.values, X_test.values, y_train.values, y_test.values
    
    def __timeseries_data_generator(self, df, timeseries_params):
        input_width = timeseries_params.input_width
        label_width = timeseries_params.label_width
        shift = timeseries_params.shift
        shuffle = timeseries_params.shuffle
        
        if any(True for output in self.output_labels if output in self.input_labels):
            input_output_label = self.input_labels
        else:
            input_output_label = self.input_labels + self.output_labels
            
        df = df[input_output_label]
        
        window_generator = WindowGenerator(input_width, label_width, shift, df, self.output_labels, shuffle)
        
        X,y = window_generator.get_train_test_data()
        
        if len(X) < self.K:
            raise ValueError('n_sample_K value can not be greater than size of input value.')
            
        X_train, y_train = X[:self.K] , y[:self.K]
        end_point_index = self.K + self.n_test_points
        
        if end_point_index > len(X):
            X_test, y_test = X[self.K:], y[self.K:]
        else:
            X_test, y_test = X[self.K:end_point_index], y[self.K:end_point_index]
        
        return X_train, X_test, y_train, y_test
        
    
    
    
        
        
        
        