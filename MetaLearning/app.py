# -*- coding: utf-8 -*-


import os
from datetime import datetime
import numpy as np

from MAML import MAML
import tensorflow as tf
import warnings
from hyperopt import Trials, STATUS_OK, rand, tpe, hp, fmin

from Config.Fetchparams import Fetchparams
from Models.Model_Factory import Model_Factory,Model_Name

from Models.Model_Space_And_Params import Model_Space_Params

from MetaLearning_Params import MetaLearning_Params

import mlflow
from mlflow import keras as mlflow_keras
import gin

if __name__ == '__main__':   
   
    gin.parse_config_file('Config\params.gin')
    
    tf.keras.backend.set_floatx('float64')
    warnings.filterwarnings(action='ignore')    
    
    params = Fetchparams()
    
    # Input - Output Data label
    input_label = params.get_input_label()
    output_label = params.get_output_label()
    
    # Meta Data Path
    meta_data_folder_path = params.get_meta_data_folder_path()
    
    epochs = params.get_epochs()
    
    #Meta Learning params
    n_samples_k = params.get_n_samples_k()
    alpha = params.get_alpha()
    beta = params.get_beta()
    n_tasks = np.array([i for i in range(len(os.listdir(meta_data_folder_path)))])
    
    n_test_points = params.get_n_test()
    
    input_labels = list(['X'])
    output_labels = list(['y'])
        
    # Space    
    model_space_params = Model_Space_Params(Model_Name.Custom_Model)
    space = model_space_params.Get_Space(params)
    
    # ML_flow
    mlflow.set_tracking_uri(params.get_mlflow_uri_path())   
    
    def loss_function(pred_y, y):
        #return tf.keras.backend.mean(tf.keras.losses.MSE(y, pred_y))
        return tf.reduce_mean(tf.keras.losses.MSE(y, pred_y)) 
    
    
    def f(space):        
        
        meta_params = MetaLearning_Params()
        meta_params.n_tasks = n_tasks
        meta_params.n_test_points = n_test_points
        meta_params.n_samples_k = n_samples_k
        meta_params.epochs = epochs
        
        meta_params.meta_data_folder_path = meta_data_folder_path
        
        meta_params.alpha = space['alpha']
        meta_params.beta = space['beta']
        
        # model params dict
        model_params_dict = model_space_params.Get_Model_Params_dict(space, params)        
             
        #experiment_id = mlflow.create_experiment(name='Meta_Learning')
        with mlflow.start_run() as mlflow_run:
            mlflow.log_param("n_samples_k", n_samples_k)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("alpha", meta_params.alpha)
            mlflow.log_param("beta", meta_params.beta)
            
            for key,value in model_params_dict.items():
                mlflow.log_param(key,value)
            
            log_path = '../logs/task_gradient/' + str(mlflow_run.info.run_id)
            
            model_factory = Model_Factory(Model_Name.Custom_Model)
            
            label_dict = dict()
            label_dict['input_label'] = input_label
            label_dict['output_label'] = output_label
            label_dict['index_label'] = None
            
            maml = MAML(model_factory.Get_Model, loss_function,meta_params, model_params_dict)
            loss, model = maml.train_MAML(log_path,label_dict)            
            
            mlflow.log_metric('loss',loss.numpy())
            
            model_path = mlflow.get_artifact_uri() + '/' + 'Model'                 
            model.save(model_path)
            #tf.saved_model.save(model,model_path)
            #model.save_weights(model_path)
            
        return {'loss': loss, 'status': STATUS_OK, 'model': model}
    
   
    
    trials = Trials()
    best = fmin(fn=f,
                space=space,
                algo=tpe.suggest,
                max_evals=1,
                trials=trials,
                rstate=np.random.RandomState(1),
                loss_threshold = 0.001)
    
  
 