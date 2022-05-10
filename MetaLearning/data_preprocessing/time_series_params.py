# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:06:41 2022

@author: lokes
"""


class TimeSeriesParams:
    
    def __init__(self):
        self.__input_width = None
        self.__label_width = None
        self.__shift = None
        self.__shuffle = None
        self.__label_columns = None
        
    @property
    def input_width(self):
        return self.__input_width
    
    @input_width.setter
    def input_width(self, value):
        self.__input_width = value
        
    @property
    def label_width(self):
        return self.__label_width
    
    @label_width.setter
    def label_width(self, value):
        self.__label_width = value
        
    @property
    def shift(self):
        return self.__shift
    
    @shift.setter
    def shift(self, value):
        self.__shift = value
        
    @property
    def shuffle(self):
        return self.__shuffle
    
    @shuffle.setter
    def shuffle(self, value):
        self.__shuffle = value    
    
    @property
    def label_columns(self):
        return self.__label_columns
    
    @label_columns.setter
    def label_columns(self, value):
        self.__label_columns = value
    