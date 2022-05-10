# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:00:20 2022

@author: lokes
"""


import os
import numpy as np  
import pandas as pd

import configparser
import ast

import tensorflow as tf
from data_preparation.german_credit_dataset_preperation import GermanDataPreperation
from data_preparation.feature_meta_data import FeatureMetaData
from data_preparation.data_pertubation import DataPertubation
from classifier.classifiermodel import ClassifierModel

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, roc_auc_score

from hyperopt import Trials, STATUS_OK, rand, tpe, hp, fmin
import mlflow
from constants import Constants
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer
from pickle import dump,load


def loss_function(y_true, y_pred, label_smoothing):
    '''
    Binary cross entropy loss

    Parameters
    ----------
    y_true : tf.Tensor
        True value of target variable.
    y_pred : tf.Tensor
        Predicted value of target variable.
    label_smoothing : float [0, 1]
        Smoothing parameter.

    Returns
    -------
    bce_loss : tf.Tensor, float32
        Value of Binary cross entropy loss.

    '''
    
    bce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing = label_smoothing, 
                                              from_logits = False))
    return bce_loss


def compute_loss(model, X, y, X_modified, label_smoothing, reg):   
    '''
    Custom loss funtion

    Parameters
    ----------
    model : keras.Sequential object
        Model under training.
    X : tf.Tensor
        Feature Vector.
    y : tf.Tensor
        Label Vector.
    X_modified : tf.Tensor
        Feature vector with perturbed values.
    label_smoothing : float [0, 1]
        Smoothing parameter.
    reg : float [0,1]
        Sensitivity loss regularization parameter.

    Returns
    -------
    loss : tf.Tensor, float32
        Sensitivity loss value.

    '''
    output_x = model(X)
    
    # Pertub optimization data
    X_actual = X_modified[0]
    X_Pertub = X_modified[1]
    
    loss = loss_function(y, output_x, label_smoothing) + reg * tf.reduce_mean(abs(model(X_actual) - model(X_Pertub)))    
    return loss


def sensitivity_gradient(model, X_train,y_train,label_smoothing):
    '''
    Gradient for sensitivity

    Parameters
    ----------
    model : keras.Sequential object
        Model under training.
    X_train : tf.Tensor
        Feature Vector.
    y_train : tf.Tensor
        Label Vector.
    label_smoothing : float [0, 1]
        Smoothing parameter.

    Returns
    -------
    gradient : tf.Tensor
        Sensitivity gradient.

    '''
    with tf.GradientTape() as sensitivity:
        sensitivity.watch(X_train)
        loss = loss_function(y_train, model(X_train),label_smoothing )
    gradient = sensitivity.gradient(loss, X_train)
    return gradient


def model_gradient(model, X_train, y_train, X_modified, label_smoothing, reg):
    '''
    Gradient for model training

    Parameters
    ----------
    model : keras.Sequential object
        Model under training.
    X_train : tf.Tensor
        Feature Vector.
    y_train : tf.Tensor
        Label Vector.
    X_modified : tf.Tensor
        Feature vector with perturbed values.
    label_smoothing : float [0, 1]
        Smoothing parameter.
    reg : float [0,1]
        Sensitivity loss regularization parameter.

    Returns
    -------
    final_gradient : tf.Tensor
        Model gradient.

    '''
    with tf.GradientTape() as tape:                
        actual_loss = compute_loss(model, X_train, y_train, X_modified, label_smoothing,  reg)                
    final_gradient = tape.gradient(actual_loss, model.trainable_variables)
    return final_gradient


def compute_test_loss(model, test_dataset, sensitivity_threshold, label_smoothing, reg):
    '''
    Computes test loss

    Parameters
    ----------
    model : keras.Sequential object
        Model under training.
    test_dataset : tf.Tensor
        Batch dataset for test.
    sensitivity_threshold : float
        Sensitivity threshold.
    label_smoothing : float [0,1]
        Smoothing parameter.
    reg : float [0,1]
        Sensitivity loss regularization parameter.

    Returns
    -------
    test_loss : tf.Tensor
        Computed test loss value.

    '''
    test_loss = 0 
    count = 0
    for X_test, y_test in test_dataset:
        sensitivity_grad = sensitivity_gradient(model, X_test,y_test,label_smoothing)
        X_modified = get_x_modified(sensitivity_grad, X_test, threshold = sensitivity_threshold)
        
        loss = compute_loss(model, X_test, y_test, X_modified, label_smoothing, reg) 
        test_loss += loss 
        count += 1
    test_loss = test_loss/count
    return test_loss    


def get_x_modified(gradient, X_train, threshold =  0.0001):
    '''
    Get modified values of feature vector

    Parameters
    ----------
    gradient : tf.Tensor
        Sensitivity gradient.
    X_train : tf.Tensor
        Feature vector.
    threshold : float, optional
        Sensitivity threshold. The default is 0.0001.

    Returns
    -------
    X_train : tf.Tensor
        Original Feature vector.
    X_modified : tf.Tensor
        Feature vector with perturbed values.

    '''
    
    X_modified = tf.identity(X_train)
    
    arg_sensitivity_greater_than_threshold = tf.where(tf.abs(gradient) > tf.constant(threshold))
    arg_values = tf.gather_nd(X_train,indices= tf.constant(arg_sensitivity_greater_than_threshold))
    
    # data pertubation
    data_pertubation_obj = DataPertubation(discretizer_dict, sensitive_features)
    
    pertub_dict = data_pertubation_obj.get_pertub_values(arg_sensitivity_greater_than_threshold.numpy(), arg_values.numpy())
    
    binary_value_indexes = [i for i in list(pertub_dict.keys()) if  len(pertub_dict[i]) == 1]
    multiple_value_indexes = [i for i in list(pertub_dict.keys()) if  len(pertub_dict[i]) != 1]
    
    binary_values = [pertub_dict[i][0] for i in binary_value_indexes]
    multiple_values = [pertub_dict[i] for i in multiple_value_indexes]
    
    # Binary value changes
    # print([[i[0],i[1]] for i in binary_value_indexes])
    
    if len(binary_value_indexes) != 0:
        X_modified = tf.tensor_scatter_nd_update(X_modified, [[i[0],i[1]] for i in binary_value_indexes],binary_values)

    # Multiple value  
    for multiple_value_index, pertub_values in zip(multiple_value_indexes,multiple_values):
        row,column = multiple_value_index[0], multiple_value_index[1]
         
        for i in pertub_values:
            actual_data = X_modified[row]
            modified_data = tf.identity(actual_data)
            modified_data = tf.tensor_scatter_nd_update(modified_data, [[column]],[i])
            
            X_train = tf.concat([X_train, tf.reshape(actual_data, [1,-1])], 0)
            X_modified = tf.concat([X_modified,  tf.reshape(modified_data,[1,-1])], 0)
    
    return (X_train,X_modified)

def get_auc_score(model, dataset):
    '''
    

    Parameters
    ----------
    model : keras.Sequential object
        Model under training.
    dataset : tf.Tensor
        Batch dataset.

    Returns
    -------
    score : float
        ROC AUC score.

    '''
    y_predict = list()
    y_actual = list()
    for X_, y_ in dataset:
        y_actual.extend(y_.numpy().squeeze())
        y_predict.extend(model(X_).numpy().squeeze())
    score = roc_auc_score(y_actual,y_predict)
    return score


def get_train_validate_dataset(X,y, batch_size=128, n_splits = 5):
    '''
    
    Yields
    ------
    tf.data.Dataset
    Train and Test data based on kFold n_splits.
    '''

    def gen():
        kf = KFold(n_splits= n_splits)
        for train_index, test_index in kf.split(np.arange(0,X.shape[0],1,dtype= np.int64)):
        
            X_train, y_train = tf.convert_to_tensor(X.loc[train_index], dtype=tf.float32), \
                tf.convert_to_tensor(y.loc[train_index], dtype=tf.float32)
                
            X_test, y_test = tf.convert_to_tensor(X.loc[test_index], dtype=tf.float32), \
                tf.convert_to_tensor(y.loc[test_index], dtype=tf.float32)
                
            y_train = tf.reshape(y_train,shape = (-1,1))
            y_test = tf.reshape(y_test,shape = (-1,1))
            
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train,y_train))
            train_dataset = train_dataset.batch(batch_size)
            
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
            test_dataset = test_dataset.batch(batch_size)
            
            yield train_dataset, test_dataset

    return gen()


def f(space):   
    '''
    Hyperopt function to minimize

    Parameters
    ----------
    space : dict
        Hyperparameter space.

    Returns
    -------
    dict
        Test loss and status.

    '''
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
    reg = space['reg']
    sensitivity_threshold = space['sensitivity_threshold']
        
    
    # Fit model
    patience = 10
    
    cv_train = list()
    cv_test = list()  
    
    cv_auc_train = list()
    cv_auc_test = list()
    
    train_auc = list()
    test_auc = list()
    
    for train_dataset, test_dataset in get_train_validate_dataset(X,y, batch_size=BATCH_SIZE, 
                                                                  n_splits = 5):
       
        classifier_instance = ClassifierModel()
        model = classifier_instance.make_custom_model_with_3_hidden_layer(input_shape,
                                               num_classes,                                               
                                               n_layer1,
                                               act_layer1,
                                               n_layer2,
                                               act_layer2,
                                               n_layer3,
                                               act_layer3,
                                               learning_rate,
                                               output_bias)
        
        
        wait = 0
        best_loss = 0
        epochs_ = list()
        for epoch in range(EPOCHS):
            
            train_loss = 0
            count = 0
            
            for X_train, y_train in train_dataset:
  
                sensitivity_grad = sensitivity_gradient(model, X_train, y_train, label_smoothing = class_weight_ratio)
                
                # get x_modified            
                X_modified = get_x_modified(sensitivity_grad,X_train, sensitivity_threshold)
                
                # model gradient
                final_gradient = model_gradient(model, X_train, y_train, X_modified,class_weight_ratio, reg)            
                model.optimizer.apply_gradients(zip(final_gradient, model.trainable_variables))
                
                loss = compute_loss(model, X_train, y_train, X_modified, class_weight_ratio, reg)
                train_loss += loss
                count += 1
        
            train_loss = train_loss/count        
            test_loss = compute_test_loss(model, test_dataset, sensitivity_threshold, class_weight_ratio, reg)
            epochs_.append(epoch)
                
            #Early Stopping
            wait += 1
            if test_loss > best_loss:
                best_loss = test_loss
                wait = 0
            if wait >= patience:
                print("---------Early Stopping---------")
                break
            
        cv_train.append(train_loss)
        cv_test.append(test_loss)
        
        train_auc.extend([get_auc_score(model,train_dataset)])
        test_auc.extend([get_auc_score(model,test_dataset)])

   
        
    cv_mean_train = np.mean(cv_train)
    cv_std_train = np.std(cv_train)
    
    cv_mean_test = np.mean(cv_test)
    cv_std_test = np.std(cv_test)
    
    #auc
    train_auc_mean = np.mean(train_auc)
    train_auc_std = np.std(train_auc)
    
    test_auc_mean = np.mean(test_auc)
    test_auc_std = np.std(test_auc)
    
    
    # log params, metric, artifacts in mlflow
    with mlflow.start_run() as mlflow_run:
        for key in space.keys():
            mlflow.log_param(key, space[key])
       
        mlflow.log_metric('cv_mean_train', cv_mean_train)           
        mlflow.log_metric('cv_std_train', cv_std_train) 
        
        mlflow.log_metric('cv_mean_test', cv_mean_test)           
        mlflow.log_metric('cv_std_test', cv_std_test)  

        mlflow.log_metric('train_auc_mean', train_auc_mean)           
        mlflow.log_metric('train_auc_std', train_auc_std) 

        mlflow.log_metric('test_auc_mean', test_auc_mean)           
        mlflow.log_metric('test_auc_std', test_auc_std) 
        
        for k,v in enumerate(epochs_):
            mlflow.log_metric('epoch', v, k)              
               
        model_path = mlflow.get_artifact_uri() + '/' + 'Model'                 
        model.save(model_path)  
        
            
    return {'loss': cv_mean_test, 'status': STATUS_OK}

if __name__ == '__main__':    
    
    # config file path
    config_path = Constants.config_path 
    
    parser = configparser.ConfigParser()
    parser.read(config_path)
    
    feature_meta_data = FeatureMetaData()
    
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
    datapreperation = GermanDataPreperation(data_path, preprocess_dump_path, config_path)

    ### Train-Test split
    X, y, kfold = datapreperation.get_stratified_kfold_data(n_splits = 5, shuffle = True)
    
    # Create Discretizer
    # TO DO: Prpose of Loan Handling
    discretizer_dict = dict()
    for feature in feature_meta_data.FEATURE_COLUMNS:
        if len(X[feature].unique()) < 5 or feature == 'PurposeOfLoan':
            discretizer_dict[feature] = tf.constant(X[feature].unique(),dtype=tf.float32)
        else:
            est = KBinsDiscretizer(encode="ordinal",strategy="quantile").fit(X[feature].values.reshape(-1,1))
            discretizer_dict[feature] = tf.constant(est.bin_edges_[0],dtype=tf.float32)

        
    ### Fixed Params
    input_shape = X.shape[-1]
    num_classes = 1

    # Initial Bias setting for imbalanced class
    output_bias = datapreperation.get_initialbias()
    
    # Sensitive Features
    sensitive_features = ['Age', 'Gender', 'Single', 'HasTelephone']
    
    # sensitive_features = feature_meta_data.FEATURE_COLUMNS

    ### Hyper Params   
    n_layer1 = np.linspace(50, 100, 6, dtype = np.int64)
    act_layer1 = ['relu','tanh', 'sigmoid']
    n_layer2 = [input_shape]
    act_layer2 = ['relu','tanh', 'sigmoid']   
    n_layer3 = [2]
    act_layer3 = ['relu','tanh', 'sigmoid'] 
    
    learning_rate = [0.005, 0.01]
    EPOCHS = [100]
    BATCH_SIZE = [64]      #[64,128]
    class_weight_ratio = np.linspace(0.1, 0.8, 5, dtype = np.float64)
    reg =  np.linspace(0.1, 0.5, 5, dtype = np.float64)
    sensitivity_threshold = [1e-8]           #np.linspace(1e-10,1e-7,8, dtype = np.float32)


    space = {'n_layer1': hp.choice('n_layer1',n_layer1),
             'act_layer1': hp.choice('act_layer1',act_layer1),
             'n_layer2': hp.choice('n_layer2',n_layer2),
             'act_layer2': hp.choice('act_layer2',act_layer2),
             'n_layer3': hp.choice('n_layer3',n_layer3),
             'act_layer3': hp.choice('act_layer3',act_layer3),
             'learning_rate': hp.choice('learning_rate',learning_rate), 
             'EPOCHS': hp.choice('EPOCHS',EPOCHS),
             'BATCH_SIZE': hp.choice('BATCH_SIZE', BATCH_SIZE),
             'class_weight_ratio': hp.choice('class_weight_ratio', class_weight_ratio),
             'reg': hp.choice('reg', reg),
             'sensitivity_threshold': hp.choice('sensitivity_threshold',sensitivity_threshold)
             }
    
    trials = Trials()
    best = fmin(fn=f,
                space=space,
                algo=tpe.suggest,
                max_evals = 25,
                trials=trials,
                rstate=np.random.RandomState(1))  
