# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:50:50 2022

@author: lokes
"""

import numpy as np
import pandas as pd
from prediction_service.model_prediction import ModelPredictionService
from AdversarialAI.zoo import ZOO

from hyperopt import Trials, STATUS_OK, rand, tpe, hp, fmin

class AdversialAttack:
    
    def __init__(self,
                 space,
                 x_input,
                 model_path, config_path,
                 decision_threshold,                 
                 box_max,
                 box_min,
                 max_iterations,
                 max_evals):
        
        self.space = space
        self.x_input = x_input
        self.input_shape = x_input.shape
        self.prediction_service = ModelPredictionService(model_path, config_path)
        self.decision_threshold = decision_threshold
        self.box_max = box_max
        self.box_min = box_min      
        
        self.max_iterations = max_iterations
        
        self.best_pertubation = None
        self.best_loss = 1e10
        
        self.trials = Trials()
        best = fmin(fn=self.f,
                    space=space,
                    algo=tpe.suggest,
                    max_evals = max_evals,
                    trials=self.trials,
                    rstate=np.random.RandomState(1))
    
    def get_best_attack(self):
        return self.best_pertubation, self.best_loss
       
    def f(self,space):
        initial_const = space['initial_const']
        confidence = space['confidence']
        learning_rate = space['learning_rate']
        
        adversial_example = ZOO(self.input_shape,
          self.prediction_service,
          self.decision_threshold,
          initial_const,
          confidence,
          learning_rate,
          self.box_max,
          self.box_min,
          self.max_iterations)

        x_pertubation, loss, iteration_count = adversial_example.attack(self.x_input)
                        
        if loss < 0:
            loss = -loss
        
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_pertubation = x_pertubation        
        
        
        return {'loss': loss, 'status': STATUS_OK}
    
    def get_best_hyperparam(self):
        best_params_index_dict = self.get_bestparam_index()
        params_dict = dict()
        for k,v in self.space.items():
            params_dict[k] = v[best_params_index_dict[k]]
        return params_dict
            
    def get_bestparam_index(self):
        best_params_index_dict = dict()
        for key in self.space.keys():
            best_params_index_dict[key] = self.trials.best_trial['misc']['vals'][key][0]
        return best_params_index_dict
    
    