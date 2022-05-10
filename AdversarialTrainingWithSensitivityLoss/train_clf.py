

import os
import numpy as np  
import pandas as pd

import configparser
import ast

import tensorflow as tf
from data_preparation.german_credit_dataset_preperation import GermanDataPreperation
from classifier.classifiermodel import ClassifierModel
from classifier.plot_utility import PlotUtility

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

from hyperopt import Trials, STATUS_OK, rand, tpe, hp, fmin
import mlflow
from constants import Constants



def f(space):    
    # Extract hyper-params from search space
    n_layer1 = space['n_layer1']
    act_layer1 = space['act_layer1']
    n_layer2 = space['n_layer2']
    act_layer2 = space['act_layer2']
    n_layer3 = space['n_layer3']
    act_layer3 = space['act_layer3']
    learning_rate = space['learning_rate']
    EPOCHS = space['EPOCHS']
    BATCH_SIZE = space['BATCH_SIZE']
    class_weight_ratio = space['class_weight_ratio']    
        
    class_weight = {0: class_weight_ratio, 1: 1 - class_weight_ratio}
    
    # Fit model
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_prc', verbose=1, 
                                                      patience=10,mode='max',
                                                      restore_best_weights=True)
    
    scores = []
    for train_index, test_index in kfold.split(X, y):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        
        classifier_instance = ClassifierModel()
        model = classifier_instance.make_model_with_3_hidden_layer(input_shape,
                                                num_classes,
                                                loss_func,
                                                metrics,
                                                n_layer1,
                                                act_layer1,
                                                n_layer2,
                                                act_layer2,
                                                n_layer3,
                                                act_layer3,
                                                learning_rate,
                                                output_bias)
        
        history = model.fit( X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs = EPOCHS,
                  callbacks=[early_stopping],
                  validation_data=(X_test, y_test),
                  verbose = 0,
                  class_weight=class_weight)
        
        scores.append(history.history['val_auc'][-1])
    
    cv_mean_score = np.mean(scores)
    cv_std_score = np.std(scores)
    
    # log params, metric, artifacts in mlflow
    if cv_mean_score > 0.6:    
        with mlflow.start_run() as mlflow_run:
            for key in space.keys():
                mlflow.log_param(key, space[key])
           
            mlflow.log_metric('CV_Score_Mean', cv_mean_score)           
            mlflow.log_metric('CV_Score_Std', cv_std_score) 
                                   
            for i in range(0,len(history.history['val_auc']),1):
                # mlflow.log_metric('train_accuracy',history.history['accuracy'][i],step = i)
                # mlflow.log_metric('val_accuracy',history.history['val_accuracy'][i],step = i)
                mlflow.log_metric('train_precision',history.history['precision'][i],step = i)
                mlflow.log_metric('val_precision',history.history['val_precision'][i],step = i)
                mlflow.log_metric('train_recall',history.history['recall'][i],step = i)
                mlflow.log_metric('val_recall',history.history['val_recall'][i],step = i)
                mlflow.log_metric('train_auc',history.history['auc'][i],step = i)
                mlflow.log_metric('val_auc',history.history['val_auc'][i],step = i)
                mlflow.log_metric('train_prc',history.history['prc'][i],step = i)
                mlflow.log_metric('val_prc',history.history['val_prc'][i],step = i)
            model_path = mlflow.get_artifact_uri() + '/' + 'Model'                 
            model.save(model_path)
            
    return {'loss': - cv_mean_score, 'status': STATUS_OK}

if __name__ == '__main__':    
    
    # config file path
    config_path = Constants.config_path 
    
    parser = configparser.ConfigParser()
    parser.read(config_path)
    
    experiment_name = ast.literal_eval(parser.get("experiment","experiment_name"))
    data_path = Constants.data_path
    preprocess_dump_path = ast.literal_eval(parser.get("preprocess_dump_path","dump_path"))
    logs_path = ast.literal_eval(parser.get("logs_path","path"))
    
    mlruns_path = logs_path + 'mlruns/'  
    
    if  os.path.exists(logs_path) ==  False:
        os.mkdir(logs_path)
    
    if   os.path.exists(mlruns_path) ==  False:
        os.mkdir(mlruns_path)
                
    # mlflow set tracking uri 
    tracking_uri_path = mlruns_path               
    mlflow.set_tracking_uri(tracking_uri_path)
    mlflow.set_experiment(experiment_name)
    
    #data preperation    
    datapreparation = GermanDataPreperation(data_path, preprocess_dump_path, config_path)

    ### Train-Test split
    X, y, kfold = datapreparation.get_stratified_kfold_data(n_splits = 5, shuffle = True)


    ### Fixed Params
    input_shape = X.shape[-1]
    num_classes = 1
    loss_func = tf.keras.losses.BinaryCrossentropy()
    metrics = [
          tf.keras.metrics.TruePositives(name='tp'),
          tf.keras.metrics.FalsePositives(name='fp'),
          tf.keras.metrics.TrueNegatives(name='tn'),
          tf.keras.metrics.FalseNegatives(name='fn'), 
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.AUC(name='auc'),
          tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]

    # Initial Bias setting for imbalanced class
    output_bias = datapreparation.get_initialbias()

    ### Hyper Params   
    n_layer1 = np.linspace(40,100,8, dtype = np.int64)
    act_layer1 = ['relu','tanh', 'sigmoid']
    n_layer2 = [input_shape]
    act_layer2 = ['relu','tanh', 'sigmoid']   
    n_layer3 = [2]
    act_layer3 = ['relu','tanh', 'sigmoid']  
    learning_rate = [0.001, 0.005, 0.01]
    EPOCHS = [100]
    BATCH_SIZE = [64,128]
    class_weight_ratio = np.linspace(0.5,0.9,8, dtype = np.float64)


    space = {'n_layer1': hp.choice('n_layer1',n_layer1),
             'act_layer1': hp.choice('act_layer1',act_layer1),
             'n_layer2': hp.choice('n_layer2',n_layer2),
             'act_layer2': hp.choice('act_layer2',act_layer2),
             'n_layer3': hp.choice('n_layer3',n_layer3),
             'act_layer3': hp.choice('act_layer3',act_layer3),
             'learning_rate': hp.choice('learning_rate',learning_rate), 
             'EPOCHS': hp.choice('EPOCHS',EPOCHS),
             'BATCH_SIZE': hp.choice('BATCH_SIZE', BATCH_SIZE),
             'class_weight_ratio': hp.choice('class_weight_ratio', class_weight_ratio)
             }
    
    trials = Trials()
    best = fmin(fn=f,
                space=space,
                algo=tpe.suggest,
                max_evals = 50,
                trials=trials,
                rstate=np.random.RandomState(1))  
