# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:06:27 2022

@author: lokes
"""

import os
from abc import ABC, abstractmethod


class BaseDataExtractor(ABC):
    
    def __init__(self, task_path_dictionary, label_dict):
        
        for task, data_path in task_path_dictionary.items():
            if not os.path.exists(data_path):
                raise FileNotFoundError('Path %s does not exists.'%data_path)
        
        self.input_label = label_dict['input_label']
        self.output_label = label_dict['output_label']
        self.index_label = label_dict['index_label']
        
        self.task_path_dictionary = task_path_dictionary
        self.task_specific_data_dictionary = self.__extract_task_data()
        
    def get_taskspecific_data(self,task_i):
        return self.task_specific_data_dictionary[task_i]
    
    def __extract_task_data(self):
        task_specific_data_dictionary = dict()
        for task,path in self.task_path_dictionary.items():
            df = self.get_input_output_data(path, task)
            task_specific_data_dictionary[task] = df
            
        return task_specific_data_dictionary
    
    @abstractmethod
    def get_input_output_data(self,path, task_id):
        pass
        