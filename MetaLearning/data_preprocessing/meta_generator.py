# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 17:32:32 2022

@author: lokes
"""
import os
import numpy as np

from data_preprocessing.data_extractor import DataExtractor
from data_preprocessing.time_series_data_extractor import TimeSeriesDataExtractor
from data_preprocessing.task_data_generator import TaskDataGenerator

class MetaGenerator:
    
    
    def __init__(self, 
                 meta_data_path, 
                 n_tasks,
                 n_samples_k,
                 n_test_points,
                 label_dict,
                 config_params,
                 isTimeSeriesData = False):
        
        if not os.path.exists(meta_data_path):
            raise FileNotFoundError('Path %s does not exists.'%meta_data_path)
        
        self.n_tasks = n_tasks
        self.task_specific_path = self.__get_taskspecific_folder_path(meta_data_path)
        self.task_path_mapping = self.__map_path_to_task()
        
        
        if isTimeSeriesData:
            self.data_extractor_instance = TimeSeriesDataExtractor(self.task_path_mapping, label_dict, config_params)
        else:
            self.data_extractor_instance = DataExtractor(self.task_path_mapping, label_dict)
            
        self.task_data_generator = TaskDataGenerator(n_samples_k, n_test_points, label_dict, isTimeSeriesData)
        
    def sample_task_generator(self, tasks_samples):
        return np.random.choice(self.n_tasks, tasks_samples)
    
    def get_taskspecific_metatraintestset(self, task_i, timeseries_params=None):
        input_output_df = self.data_extractor_instance.get_taskspecific_data(task_i)
        X_train, X_test, y_train, y_test = self.task_data_generator.train_test_generator(input_output_df,
                                                                                         timeseries_params)
        return X_train, X_test, y_train, y_test
    
    def __map_path_to_task(self):
        task_path_mapping = dict()
        for task, path in zip(self.n_tasks, self.task_specific_path):
            task_path_mapping[task] = path
        return task_path_mapping
    
    def __get_taskspecific_folder_path(self, meta_data_path):
        task_specific_path_list = list()
        for i in os.listdir(meta_data_path):
            meta_task_path = os.path.join(meta_data_path, i)
            for j in os.listdir(meta_task_path):
                task_specific_path_list.append(os.path.join(meta_task_path,j))
        return task_specific_path_list
        
        
