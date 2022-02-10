# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 07:34:21 2022

@author: lokes
"""


from datetime import datetime
import tensorflow as tf
import numpy as np


class Logger:
    
    def __init__(self, logpath, metric_name, sub_folder_name):
        self.metric_name = metric_name
        self.sub_folder_name = sub_folder_name
        
        self.metric = self.metric()
        self.summary_writer = self.writter() 
        
    def metric(self):
        metric = tf.keras.metrics.Mean(self.metric_name, dtype=tf.float32)
        return metric  
    
    def writter(self):        
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = self.log_path + current_time + '/' + self.sub_folder_name    
        summary_writer = tf.summary.create_file_writer(log_dir)        
        return summary_writer
    
    def register_log(self,value):
        self.metric(value)
        
    def log_Metric(self, step):
        with self.summary_writer.as_default():
            tf.summary.scalar(self.metric_name, self.metric.result(), step=step)
            
    def reset(self):
        self.metric.reset_states()    
        
        