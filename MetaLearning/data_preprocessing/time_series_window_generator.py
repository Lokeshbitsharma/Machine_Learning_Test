# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 07:49:29 2022

@author: lokes
"""

import numpy as np
import pandas as pd
import tensorflow as tf

class WindowGenerator:
    
    def __init__(self,input_width, label_width, shift, train_df, label_columns, shuffle=False):
        
        self.train_df = train_df
        self.shuffle = shuffle
        
        self.label_columns = label_columns
        
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
            
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        self.input_slice = slice(0,input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
    def __repr__(self):
        return '\n'.join([
            f'Total Window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    def split_window(self, features):
        inputs = features[:, self.input_slice,:]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:,:,self.column_indices[name]] for name in self.label_columns], axis = -1)
            
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    def make_dataset(self, data):
        data = np.array(data, dtype = np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(data= data,
                                                                  targets= None,
                                                                  sequence_length = self.total_window_size,
                                                                  sequence_stride = 1,
                                                                  shuffle = self.shuffle,
                                                                  batch_size = 128,)
        ds = ds.map(self.split_window)
        return ds
    
    def get_train_test_data(self):
        X, y = list(),list()
        for input_, label_ in self.train.take(1):
            X.append(input_.numpy())
            y.append(label_.numpy())
        return (X[0],y[0])
    