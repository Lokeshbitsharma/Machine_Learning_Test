# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:03:18 2022

@author: lokes
"""



import numpy as np

import tensorflow as tf 

from Models.Stacked_Model import Stacked_Model
from Config.Fetchparams import Fetchparams

from Data_Preprocessing.TimeSeries_WindowGenerator import TimeSeries_Data_Params

from Adapt.TimeSeries_Meta_Adaptation import TimeSeries_Meta_Adaptation
from TimeSeriesEvaluation.TimeSeries_Model_Evaluation import Model_Evaluation

from sklearn.model_selection import train_test_split
from hyperopt import Trials, STATUS_OK, rand, tpe, hp, fmin

import mlflow
import joblib
import gin

gin.parse_config_file('Config\params_TimeSeries.gin')

file_path = 'Time_Series_Data/Train/HEALTH/HEALTH_WITH_EPIDEMIC.csv'
mlflow_uri_path = '../Meta_Adapt_logs/MFG/ML_Flow/mlruns'
scaler_path = 'Saved_Models/Revenue in Mn USD_HEALTH.pkl'


# Task Path dictionary
task_path_dictionary = dict()
task_path_dictionary[0] = file_path


# Fetch config params and time series params
params = Fetchparams()

label_dict = dict()
label_dict['input_label'] = params.get_input_label()
label_dict['output_label'] = params.get_output_label()
label_dict['index_label'] = params.get_time_index_label() 

#Time series params
timeSeries_params = TimeSeries_Data_Params()
timeSeries_params.input_width = 6
timeSeries_params.label_width = 3
timeSeries_params.shift = 3
timeSeries_params.shuffle = False
timeSeries_params.label_columns = label_dict['output_label']


# Loss Function
def loss_function(pred_y, y):
    return tf.reduce_mean(tf.keras.losses.MSE(y, pred_y)) 


# Set mlflow uri path
mlflow.set_tracking_uri(mlflow_uri_path) 


# Retrieve_Scaler
retrieve_scaler = False

### Get X,y and Train test data
model_evaluation = Model_Evaluation(label_dict, params, timeSeries_params, task_path_dictionary,retrieve_scaler)

X,y = model_evaluation.Get_Input_Output_Data(0)
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30,random_state=42)

# Load scaler 
std_scaler = joblib.load(scaler_path)


### Hyper-params search space

unfreeze_number_of_layers = [0,1,2]

neurons_l1 = np.linspace(40,60,4,dtype=np.int32)

neurons_out = 1
activation_l1 = ['tanh','relu']
activation_l2 = ['tanh','relu']

optimizer = ['Adam']
learning_rate = [0.01,0.005,0.001]
epochs = [50,75,100,125]




space = {'unfreeze_number_of_layers': hp.choice('unfreeze_number_of_layers',unfreeze_number_of_layers),
          'neurons_l1': hp.choice('neurons_l1',neurons_l1),  
          'activation_l1': hp.choice('activation_l1', activation_l1),
          'activation_l2': hp.choice('activation_l2',activation_l2),
         'optimizer': hp.choice('optimizer',optimizer),
         'learning_rate': hp.choice('learning_rate',learning_rate),
         'epochs': hp.choice('epochs',epochs)}


def f(space): 
    
    # Meta Model 
    model_path = '../logs/ML_Flow/mlruns/0/824c866921e249f1a0d45c7d53d060a0/artifacts/Model'
    model_custom_objects = {"Stacked_Model": Stacked_Model}
        
    unfreeze_number_of_layers = space['unfreeze_number_of_layers']    
    
    meta_adapt_params_dict = dict()
    meta_adapt_params_dict['input_shape'] = (timeSeries_params.label_width, 1)
    meta_adapt_params_dict['neurons_l1'] = space['neurons_l1']
    meta_adapt_params_dict['neurons_out'] = neurons_out
    meta_adapt_params_dict['activation_l1'] =  space['activation_l1']
    meta_adapt_params_dict['activation_l2'] =  space['activation_l2']
    
    optimizer = space['optimizer']
    learning_rate = space['learning_rate']   
    epochs = space['epochs']
    
    ## Train Meta model
    meta_adapt = TimeSeries_Meta_Adaptation(model_path,
                                            model_custom_objects,
                                            unfreeze_number_of_layers,                
                                            meta_adapt_params_dict,
                                            loss_function,
                                            optimizer,
                                            learning_rate,
                                            epochs)
        
        
        
    train_loss_list, test_loss_list, meta_adapt_model = meta_adapt.train_with_sampling(X,y)
    
    with mlflow.start_run() as mlflow_run:
        
        mlflow.log_param("unfreeze_number_of_layers", unfreeze_number_of_layers)
        mlflow.log_param("input_shape", meta_adapt_params_dict['input_shape'])
        mlflow.log_param("neurons_l1",  meta_adapt_params_dict['neurons_l1'])
        mlflow.log_param("neurons_out", meta_adapt_params_dict['neurons_out'])
        mlflow.log_param("activation_l1", meta_adapt_params_dict['activation_l1'])
        mlflow.log_param("activation_l2", meta_adapt_params_dict['activation_l2'])
        mlflow.log_param("epochs", epochs)
        
        for i in range(epochs):
            mlflow.log_metric("train_loss", train_loss_list[i],step=i)
            mlflow.log_metric("test_loss", test_loss_list[i],step=i)
        
        model_path = mlflow.get_artifact_uri() + '/' + 'Model' 
        meta_adapt_model.save(model_path)

    return {'loss': test_loss_list[-1], 'status': STATUS_OK, 'model': meta_adapt_model}


trials = Trials()
best = fmin(fn=f,
            space=space,
            algo=tpe.suggest,
            max_evals=80,
            trials=trials,
            rstate=np.random.RandomState(1),
            loss_threshold = 0.05)
