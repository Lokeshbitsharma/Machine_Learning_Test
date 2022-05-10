# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:49:24 2022

@author: lokes
"""


class AttackedInstance:
    
    def __init__(self):
        self.__purtub_input = None
        self.__actual_input = None
        self.__purtub_score = None
        self.__actual_score = None
        self.__loss = None   
        self.__index = None
        
    @property    
    def purtub_input(self):
        return self.__purtub_input
    
    @purtub_input.setter
    def purtub_input(self,value):
        self.__purtub_input = value
         
    @property    
    def actual_input(self):
        return self.__actual_input
    
    @actual_input.setter
    def actual_input(self,value):
        self.__actual_input = value
        
    @property    
    def purtub_score(self):
        return self.__purtub_score
    
    @purtub_score.setter
    def purtub_score(self,value):
        self.__purtub_score = value
        
    @property    
    def actual_score(self):
        return self.__actual_score
    
    @actual_score.setter
    def actual_score(self,value):
        self.__actual_score = value
        
    @property    
    def loss(self):
        return self.__loss
    
    @loss.setter
    def loss(self,value):
        self.__loss = value
        
    @property    
    def index(self):
        return self.__index
    
    @index.setter
    def index(self,value):
        self.__index = value
        