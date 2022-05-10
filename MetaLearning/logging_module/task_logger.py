# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:44:56 2022

@author: lokes
"""

from datetime import datetime
from logging_module.base_logger import BaseLogger

import tensorflow as tf
import numpy as np

class TaskLogger(BaseLogger):
    
    def __init__(self,n_tasks, log_path, metric_name, subfolder_name):
        self.metric_name = metric_name
        self.subfolder_name = subfolder_name
        
        super().__init__(n_tasks, log_path)
        
        self.metric_dict = self.task_metric()
        self.summary_writter_dict = self.task_writter()
        
    def task_metric(self):
        metric_dict = dict()
        for i in self.n_tasks:
            _ = self.metric_name + '_' + str(i)
            
            metric_dict[i] = tf.keras.metric.Mean(_,dtype=tf.float32)
            
        return metric_dict
    
    def task_writter(self):
        summary_writter_dict = dict()
        for i in self.n_tasks:
            current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_dir = self.log_path + current_time + '/' + self.subfolder_name + '_' + str(i)
            summary_writter = tf.summary.create_file_writer(log_dir)
            summary_writter_dict[i] = summary_writter
        return summary_writter_dict
    
    def register_log(self, task_id, value):
        self.metric_dict[task_id](value)
        
    def log_metric(self, task_id, step):
        with self.summary_writter_dict[task_id].as_default():
            tf.summary.scalar(self.metric_name + '_' + str(task_id), self.metric_dict[task_id].result(),step=step)
            
    def reset(self):
        for k,v in self.metric_dict.items():
            v.reset_states()
            


