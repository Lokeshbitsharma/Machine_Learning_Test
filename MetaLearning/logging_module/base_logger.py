# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:13:51 2022

@author: lokes
"""

import os
from abc import abstractmethod, ABC

class BaseLogger(ABC):
    
    def __init__(self, n_tasks, log_path):
        self.n_tasks = n_tasks
        self.log_path = log_path
        
    @abstractmethod
    def task_metric(self):
        pass
    
    @abstractmethod
    def task_writter(self):
        pass