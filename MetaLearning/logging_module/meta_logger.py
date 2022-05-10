# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:17:27 2022

@author: lokes
"""

from datetime import datetime
from logging_module.base_logger import BaseLogger

import tensorflow as tf
import numpy as np

class MetaLogger(BaseLogger):
    
    def __init__(self, log_path, metric_name, subfolder_name):
        self.metric_name = metric_name
        self.subfolder_name = subfolder_name
        
        n_tasks = np.array([1])
        super().__init__(n_tasks, log_path)
        
        self.metric = self.task_metric()
        self.summary_writter = self.task_writter()
        
    def task_metric(self):
        metric = tf.keras.metric.Mean(self.metric_name, dtype = tf.float32)
        return metric
    
    def task_writter(self):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.log_path + current_time + '/' + self.subfolder_name
        summary_writter = tf.summary.create_file_writer(log_dir)
        return summary_writter
    
    def register_log(self,value):
        self.metric(value)
        
    def log_metric(self,step):
        with self.summary_writter().as_default():
            tf.summary.scalar(self.metric_name, self.metric.result(), step = step)
    
    def reset(self):
        self.metric.reset_states()
        