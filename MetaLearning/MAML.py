# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 13:00:05 2022

@author: lokes
"""

import time
import tensorflow as tf

from logging_module.task_logger import TaskLogger
from logging_module.meta_logger import MetaLogger
from data_preprocessing.meta_generator import MetaGenerator



class MAML:
    
    def __init__(self,
                 model_func,
                 loss_func,
                 meta_params,
                 model_params,
                 time_series_params =None,
                 isTimeseriesData = False):
        
        self.model_func = model_func
        self.loss_func = loss_func
        
        self.n_tasks = meta_params.n_tasks
        self.n_samples_k = meta_params.n_samples_k
        self.n_test_points = meta_params.n_test_points
        self.epochs = meta_params.epochs
        self.alpha = meta_params.alpha
        self.beta = meta_params.beta
        
        self.meta_data_folder_path = meta_params.meta_data_folder_path
        self.model_params = model_params
        
        self.time_series_params = time_series_params
        self.isTimeseriesData = isTimeseriesData
        self.model = model_func(self.model_params_dict)
    
    def compute_loss(self, model, X, y):
        model_output = model(X)
        loss = self.loss_func(y, model_output)
        return loss, model_output
    
    def copy_model(self, X):
        copied_model = self.model_func(self.model_params)
        copied_model(X)
        copied_model.set_weights(self.model.get_weights())
        return copied_model
    
    def train_MAML(self, log_path, label_dict, config_params=None):
        
        optimizer = tf.keras.optimizer.Adam(learning_rate = self.beta)
        
        task_logger = TaskLogger(self.n_tasks, log_path = log_path,
                                             metric_name = 'train_loss',
                                             subfolder_name = 'train')
        
        meta_logger = MetaLogger(log_path= log_path,
                                             metric_name = 'meta_loss',
                                             subfolder_name = 'meta_learning')
        
        meta_generator = MetaGenerator(self.meta_data_folder_path,
                                       self.n_tasks,
                                       self.n_samples_k,
                                       self.n_test_points,
                                       label_dict,
                                       config_params,
                                       self.isTimeseriesData)
        
        for e in range(self.epochs):
            start = time.time()
            total_loss = 0
            lossess = []
            meta_gradient = None
            
            for i in self.n_tasks:
                X_train, X_test, y_train, y_test = meta_generator.get_taskspecific_metatraintestset(i, self.time_series_params)
                
                with tf.GradientTape() as test_tape:
                    with tf.GradientTape() as train_tape:
                        train_loss, y_hat_train = self.compute_loss(self.model, X_train, y_train)
                    task_gradients = train_tape.gradient(train_loss, self.model.trainable_variables)
                    
                    
                    task_logger.register_log(task_id = i, value = train_loss)
                    task_logger.log_metric(task_id = i, step = e)
                    
                    model_copy = self.copy_model(X_train)
                    
                    # Update inner gradients
                    k = 0
                    for j in range(len(model_copy.trainable_variables)):
                        model_copy.trainable_variables[j].assign(tf.subtract(self.model.trainable_variables[j], tf.multiply(self.alpha, task_gradients[j])))
                    k += 2
                    
                    test_loss, y_hat_test = self.compute_loss(model_copy, X_test, y_test)
                    
                if meta_gradient is None:
                    meta_gradient = test_tape.gradient(test_loss, model_copy.trainable_variables)
                else:
                    meta_gradient += test_tape.gradient(test_loss, model_copy.trainable_variables)
                    
                total_loss += test_loss
                loss = total_loss/(i + 1.0)
                
            lossess.append(loss)
            optimizer.apply_gradients(zip(meta_gradient, self.model.trainable_variables))
            
            divisor = 5 if self.epochs<100 else 50
            
            if (e+1) % divisor == 0:
                print('Epoch {}: loss = {}, Time to run {}'.format(e+1, loss, time.time() - start))
                
            meta_logger.register_log(loss)
            meta_logger.log_metric(step = e)
            
            task_logger.reset()
            meta_logger.reset()
        
        return loss, self.model           
                    