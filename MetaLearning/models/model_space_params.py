# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from Models.Model_Factory import Model_Name
from hyperopt import hp

class Model_Space_Params:
    
    def __init__(self, model_name):
        self.model_name = model_name
            
    def Get_Space(self,params):
        
        
        if self.model_name is Model_Name.Custom_Model:
            alpha = params.get_alpha()
            beta = params.get_beta()
            neurons_l1 = params.get_neurons_l1_range()
            neurons_l2 = params.get_neurons_l2_range()
            neurons_out = params.get_neurons_out()
            act_l1 = params.get_activation_l1()
            act_l2 = params.get_activation_l2() 
        
            space = {'alpha': hp.choice('alpha',alpha),
                     'beta': hp.choice('beta',beta),
                     'neurons_l1': hp.choice('neurons_l1',neurons_l1), 
                     'neurons_l2': hp.choice('neurons_l2',neurons_l2),
                     'neurons_out': hp.choice('neurons_out', neurons_out),
                     'act_l1': hp.choice('act_l1', act_l1),
                     'act_l2': hp.choice('act_l2',act_l2)}
            
            return space
        
        elif self.model_name is Model_Name.Vanilla_LSTM_Model:
            alpha = params.get_alpha()
            beta = params.get_beta()
            
            neurons_l1 = params.get_neurons_l1_range()
            neurons_out = params.get_output_width()
            act_l1 = params.get_activation_l1()
            dropout_l1 = params.get_dropout_l1()
            isdropout = params.get_is_dropout()
            
            input_width = params.get_input_width()
            label_width = params.get_output_width()
            shift = params.get_shift()
            shuffle = params.get_shuffle()
            
            space = {'alpha': hp.choice('alpha',alpha),
                     'beta': hp.choice('beta',beta),             
                     'neurons_l1': hp.choice('neurons_l1',neurons_l1), 
                     'neurons_out': hp.choice('neurons_out', neurons_out),
                     'dropout_l1' : hp.choice('dropout_l1', dropout_l1),
                     'act_l1': hp.choice('act_l1', act_l1),
                     'isDropout': hp.choice('isDropout',isdropout),
                     'input_width': hp.choice('input_width',input_width),
                     'label_width': hp.choice('label_width',label_width),
                     'shift': hp.choice('shift',shift),
                     'shuffle': hp.choice('shuffle',shuffle)
             }
            
            return space
        
        elif self.model_name is Model_Name.Stacked_LSTM_Model:
            
            alpha = params.get_alpha()
            beta = params.get_beta()
            
            neurons_l1 = params.get_neurons_l1_range()
            neurons_l2 = params.get_neurons_l2_range()            
            neurons_out = params.get_output_width()
            
            act_l1 = params.get_activation_l1()
            act_l2 = params.get_activation_l2()
            act_l3 = params.get_activation_l3()
            
            input_width = params.get_input_width()
            label_width = params.get_output_width()
            shift = params.get_shift()
            shuffle = params.get_shuffle()
            
            space = {'alpha': hp.choice('alpha',alpha),
                     'beta': hp.choice('beta',beta),             
                     'neurons_l1': hp.choice('neurons_l1',neurons_l1),
                     'neurons_l2': hp.choice('neurons_l2',neurons_l2),                     
                     'neurons_out': hp.choice('neurons_out', neurons_out),                     
                     'act_l1': hp.choice('act_l1', act_l1),
                     'act_l2': hp.choice('act_l2', act_l2),
                     'act_l3': hp.choice('act_l3',act_l3),
                     'input_width': hp.choice('input_width',input_width),
                     'label_width': hp.choice('label_width',label_width),
                     'shift': hp.choice('shift',shift),
                     'shuffle': hp.choice('shuffle',shuffle)
             }
            
            return space
        
        else:
            raise ValueError('Model Name is not valid.')
        
    
    def Get_Model_Params_dict(self,space,params, timeSeries_params = None):
        
        if self.model_name is Model_Name.Custom_Model:
            
            input_shape = params.get_input_shape()
            
            model_params_dict = dict()
            model_params_dict['input_shape'] = input_shape
            model_params_dict['neurons_l1'] = space['neurons_l1']
            model_params_dict['neurons_l2'] = space['neurons_l2']
            model_params_dict['neurons_out'] = space['neurons_out']
            model_params_dict['act_l1'] = space['act_l1']
            model_params_dict['act_l2'] = space['act_l2']
            
            return model_params_dict
        
        elif self.model_name is Model_Name.Vanilla_LSTM_Model:
            
            input_label = params.get_input_label()
            
            if timeSeries_params is None:
                raise ValueError('Time series params can not be None.')
            
            
            model_params_dict = dict()
            model_params_dict['input_width'] = timeSeries_params.input_width
            model_params_dict['n_features'] = len(input_label)
            model_params_dict['neurons_l1'] = space['neurons_l1']
            model_params_dict['neurons_out'] = space['neurons_out']
            model_params_dict['dropout_l1'] = space['dropout_l1']
            model_params_dict['act_l1'] = space['act_l1']
            model_params_dict['isDropout'] = space['isDropout']
            
            return model_params_dict
        
        elif self.model_name is Model_Name.Stacked_LSTM_Model:
            
            input_label = params.get_input_label()
            
            if timeSeries_params is None:
                raise ValueError('Time series params can not be None.')
                
            model_params_dict = dict()
            model_params_dict['input_width'] = timeSeries_params.input_width
            model_params_dict['n_features'] = len(input_label)
            model_params_dict['neurons_l1'] = space['neurons_l1']
            model_params_dict['neurons_l2'] = space['neurons_l2']
           
            model_params_dict['neurons_out'] = space['neurons_out']
           
            model_params_dict['act_l1'] = space['act_l1']
            model_params_dict['act_l2'] = space['act_l2']
            model_params_dict['act_l3'] = space['act_l3']
            
            return model_params_dict
        
        else:
            raise ValueError('Model Name is not valid.')
        
