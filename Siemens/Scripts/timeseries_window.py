# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 19:56:37 2022

@author: lokes
"""

import numpy as np
import pandas as pd


class WindowGenerator:
    
    def __init__(self, input_width):
        self.input_width = input_width
        
    def get_train_data(self, data_frame):
        if data_frame.shape[0] < self.input_width:
            raise ValueError('Input width is greater than size of the number of data points in frame')
        
        input_list = list()
        for i in range(data_frame.shape[0]):
            
            if i <= (data_frame.shape[0] - self.input_width):
                input_list.append(data_frame.iloc[i : i + self.input_width ].values)
        
        return input_list
                
        