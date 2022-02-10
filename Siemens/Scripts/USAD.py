# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 23:15:17 2022

@author: lokes
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import activations 
from timeseries_window import WindowGenerator
from logger import Logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneOut,KFold

from hyperopt import Trials, STATUS_OK, rand, tpe, hp, fmin

import mlflow


def get_encoder_hiddenlayer_neurons_list(input_feature, latent_dim): 
    '''
    This method provides list of neurons for encoder hidden layer based on input feature count and latent dimension count
    Parameters
    ----------
    input_feature : int
        DESCRIPTION.
    latent_dim : int
        DESCRIPTION.

    Returns
    -------
    hiddenlayer_neurons_list : list
        List of neurons in encoder hidden layers.

    '''
    hiddenlayer_neurons_list = list()
    current_dim = int(input_feature/2)
    while current_dim > latent_dim:
        hiddenlayer_neurons_list.append(current_dim)
        current_dim = int(current_dim/2)
    hiddenlayer_neurons_list.append(latent_dim)
    return hiddenlayer_neurons_list
       
def get_decoder_hiddenlayer_neurons_list(input_feature, latent_dim):    
    '''
    This method provides list of neurons for decoder hidden layer based on input feature count and latent dimension count

    Parameters
    ----------
    input_feature : int
        DESCRIPTION.
    latent_dim : int
        DESCRIPTION.

    Returns
    -------
    hiddenlayer_neurons_list : List
        List of neurons in encoder hidden layers.
    '''
    hiddenlayer_neurons_list = list()
    current_dim = int(latent_dim * 2)
    
    while current_dim < input_feature:
        hiddenlayer_neurons_list.append(current_dim)
        current_dim = int(2*current_dim)
    hiddenlayer_neurons_list.append(input_feature)
    return hiddenlayer_neurons_list
    
        
def encoder(hidden_unit_list):
    '''    
    This method returns encoder model
    Parameters
    ----------
    hidden_unit_list : list
        DESCRIPTION.

    Returns
    -------
    model : Model
        Encoder Network.

    '''
    model = tf.keras.Sequential()
    
    for unit in hidden_unit_list:
        model.add(tf.keras.layers.Dense(unit, activation="relu"))
         
    return model

def decoder(hidden_unit_list):
    '''    
    This method returns decoder model
    Parameters
    ----------
    hidden_unit_list : list
        DESCRIPTION.

    Returns
    -------
    model : Model
        Decoder Network..

    '''
    model = tf.keras.Sequential()
    
    for unit in hidden_unit_list:
        model.add(tf.keras.layers.Dense(unit, activation="relu"))
   
    return model

@tf.function
def compute_loss(encoder, decoder_1,decoder_2, input_batch,n):  
    '''
    This method computes AE1 and AE2 losser for USAD.

    Parameters
    ----------
    encoder : Model
        DESCRIPTION.
    decoder_1 : Model
        DESCRIPTION.
    decoder_2 : Model
        DESCRIPTION.
    input_batch : [batch, input_width, features]
        DESCRIPTION.
    n : int
        Epoch count.

    Returns
    -------
    loss1 : float
        AE1 loss.
    loss2 : float
        AE2 loss.

    '''
    z = encoder(input_batch)
    w1 = decoder_1(z)
    w2 = decoder_2(z)
    w3 = decoder_2(encoder(w1))
    
    loss1 = 1/n * tf.reduce_mean((input_batch-w1)**2) + (1-1/n)* tf.reduce_mean((input_batch-w3)**2)        
    loss2 = 1/n * tf.reduce_mean((input_batch-w2)**2) - (1-1/n) * tf.reduce_mean((input_batch-w3)**2)
    
    return loss1,loss2

@tf.function
def loss_AE1(encoder, decoder_1, decoder_2, input_batch, epoch, optimizer1_var_list):
    '''
    AE1 Loss.
    Parameters
    ----------
    encoder : Model
        DESCRIPTION.
    decoder_1 : Model
        DESCRIPTION.
    decoder_2 : Model
        DESCRIPTION
    input_batch : [batch, input_width, features]
        DESCRIPTION.
    epoch : int
        Epoch count.
    optimizer1_var_list : list
        DESCRIPTION.

    Returns
    -------
    loss1_grads : tensor
        DESCRIPTION.
    loss1 : float
        DESCRIPTION.
    loss2 : float
        DESCRIPTION.

    '''
    with tf.GradientTape() as tape:
        loss1,loss2 = compute_loss(encoder, decoder_1, decoder_2, input_batch, epoch+1)
    loss1_grads = tape.gradient(loss1, optimizer1_var_list)
    
    return loss1_grads, loss1,loss2

@tf.function
def loss_AE2(encoder, decoder_1, decoder_2, input_batch, epoch,optimizer2_var_list):
    '''
    AE2 Loss.

    Parameters
    ----------
    encoder : Model
        DESCRIPTION.
    decoder_1 : Model
        DESCRIPTION.
    decoder_2 : Model
        DESCRIPTION
    input_batch : [batch, input_width, features]
        DESCRIPTION.
    epoch : int
        Epoch count.
    optimizer2_var_list : tensor
        DESCRIPTION.

    Returns
    -------
    loss2_grads : tensor
        DESCRIPTION.
    loss1 : float
        DESCRIPTION.
    loss2 : float
        DESCRIPTION.

    '''
    with tf.GradientTape() as tape:
        loss1,loss2 = compute_loss(encoder, decoder_1, decoder_2, input_batch, epoch+1)
    loss2_grads = tape.gradient(loss2, optimizer2_var_list)
    return loss2_grads,loss1,loss2


def train(epochs, encoder, decoder_1, decoder_2, dataset, learning_rate1, learning_rate2, batch_size = 128, n_splits = 5):
    '''
    Train USAD.

    Parameters
    ----------
    epochs : int
        DESCRIPTION.
    encoder : Model
        DESCRIPTION.
    decoder_1 : Model
        DESCRIPTION.
    decoder_2 : Model
        DESCRIPTION.
    dataset : tf.data.Dataset
        DESCRIPTION.
    learning_rate1 : float
        DESCRIPTION.
    learning_rate2 : float
        DESCRIPTION.
    batch_size : int, optional
        DESCRIPTION. The default is 128.
    n_splits : int, optional
        DESCRIPTION. The default is 5.

    Returns
    -------
    mean_score : float
        DESCRIPTION.
    std_score : float
        DESCRIPTION.

    '''
    
    # Optimizers for AE1 and AE2    
   
    cv_val_loss_1 = list()
    cv_val_loss_2 = list()
    
    for train_data, validate_data in dataset:   
        
        optimizer1 = tf.keras.optimizers.Adam(learning_rate=learning_rate1)
        optimizer2 = tf.keras.optimizers.Adam(learning_rate=learning_rate2)       
            
    
        for epoch in range(epochs):       
            start = time.time()
            total_loss_1 = 0
            total_loss_2 = 0
            count = 0
            
            
            for input_batch in train_data.as_numpy_iterator():
                
                if epoch == 0:
                    z = encoder(input_batch)
                    w1 = decoder_1(z)
                    w2 = decoder_2(z)
                
                optimizer1_var_list =  encoder.trainable_variables + decoder_1.trainable_variables
                optimizer2_var_list = encoder.trainable_variables + decoder_2.trainable_variables            
                
                #Train AE1:       
                loss1_grads, loss1, loss2 = loss_AE1(encoder, decoder_1, decoder_2, input_batch, epoch,optimizer1_var_list)            
                
                #Train AE2           
                loss2_grads, loss1, loss2 = loss_AE2(encoder, decoder_1, decoder_2, input_batch, epoch,optimizer2_var_list)
                
                optimizer1.apply_gradients(zip(loss1_grads, optimizer1_var_list))
                optimizer2.apply_gradients(zip(loss2_grads, optimizer2_var_list))
                
                total_loss_1 += loss1
                total_loss_2 += loss2
                count += 1
            total_loss_1 = total_loss_1/count
            total_loss_2 = total_loss_2/count
            
            # validation Loss 
            total_val_loss1 = 0
            total_val_loss2 = 0
            count = 0
            for val_batch in validate_data.as_numpy_iterator():
                val_loss1, val_loss2  = compute_loss(encoder, decoder_1, decoder_2, val_batch, epoch+1)
                total_val_loss1 += val_loss1
                total_val_loss2 += val_loss2
                count += 1
                
            total_val_loss1 = total_val_loss1 / count
            total_val_loss2 = total_val_loss2 / count
                
            #TO DO: batch wise val_loss normalization
            # val_loss_1, val_loss_2 = compute_loss(encoder, decoder_1, decoder_2, input_batch,epoch+1)
            
            if (epoch+1)%5 == 0:
                    print('Epoch {}: ,train_loss_1 = {}, train_loss_2 = {},Time to run: {}'.format(epoch+1, total_loss_1, total_loss_2, time.time() - start))
                    print('Epoch {}: ,val_loss_1 = {}, val_loss_2 = {},Time to run: {}'.format(epoch+1, total_val_loss1, total_val_loss2, time.time() - start))
        
        
        cv_val_loss_1.append(total_val_loss1)
        cv_val_loss_2.append(total_val_loss2)
        
    mean_score = (np.mean(cv_val_loss_1), np.mean(cv_val_loss_2))
    std_score = (np.std(cv_val_loss_1), np.std(cv_val_loss_2))
                
    return mean_score, std_score
        
    
def get_time_series_window_data(cluster_df_dict_basedon_hour_gap, min_max_scaler, input_width, n_splits = 3):
    '''
    This method creat dataset based on input_width and creates data list.

    Parameters
    ----------
    cluster_df_dict_basedon_hour_gap : dict
        DESCRIPTION.
    min_max_scaler : TYPE
        DESCRIPTION.
    input_width : int
        DESCRIPTION.
    n_splits : int, optional
        DESCRIPTION. The default is 3.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    window_generated_data_list = list()
    for i in list(cluster_df_dict_basedon_hour_gap.keys()):
        if  cluster_df_dict_basedon_hour_gap[i].shape[0]>=input_width :
            gen = WindowGenerator(input_width)

            scaled_cluster_data = cluster_df_dict_basedon_hour_gap[i][feature_column].copy()
            scaled_cluster_data[feature_column] = min_max_scaler.transform(cluster_df_dict_basedon_hour_gap[i][feature_column])        
            scaled_cluster_data.index = cluster_df_dict_basedon_hour_gap[i].index

            window_generated_data_list.extend(gen.get_train_data(scaled_cluster_data))

    return np.array(window_generated_data_list)

def get_train_validate_dataset(data, batch_size=128, n_splits = 5):
    '''
    
    Yields
    ------
    tf.data.Dataset
        Train and Test data based on kFold n_splits.

    '''
    
    def gen():
        kf = KFold(n_splits= n_splits)    
        for train_index, test_index in kf.split(np.arange(0,data.shape[0],1,dtype= np.int64)):            
            
            train, test = tf.convert_to_tensor(data[train_index], dtype=tf.float32) , tf.convert_to_tensor(data[test_index], dtype=tf.float32) 
                       
            train_dataset = tf.data.Dataset.from_tensor_slices(train)
            train_dataset = train_dataset.batch(batch_size)
            
            test_dataset = tf.data.Dataset.from_tensor_slices(test)
            test_dataset = test_dataset.batch(batch_size)
            
            yield train_dataset, test_dataset
        
    return gen()

            
            
if __name__ == "__main__":
    
    # Load Data
    data_v1 = pd.read_csv('../data/CurrentVoltage_V1.csv')
    feature_column = ['VL1', 'VL2', 'VL3', 'IL1', 'IL2', 'IL3', 'VL12',
       'VL23', 'VL31', 'INUT']
    
    
    # Scale data
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit_transform(data_v1[feature_column])
    
    
    # create clusters splitting index where hour greater than 1 hr
    index_hour_greater_than_1 = data_v1[data_v1.data_collection_frequency/np.timedelta64(1,'h') >= 1].index
    cluster_df_dict_basedon_hour_gap = dict()
    for i, index_ in enumerate(index_hour_greater_than_1 ):
        if i+1 < len(index_hour_greater_than_1):
            cluster_name = 'cluster_' + str(i)
            if i == 0:
                
                cluster_df_dict_basedon_hour_gap[cluster_name] = data_v1.loc[:index_hour_greater_than_1[i]]
                cluster_df_dict_basedon_hour_gap[cluster_name] = cluster_df_dict_basedon_hour_gap[cluster_name].drop([index_hour_greater_than_1[i]],0)
            else:
                cluster_df_dict_basedon_hour_gap[cluster_name] = data_v1.loc[index_hour_greater_than_1[i]:index_hour_greater_than_1[i+1]] 
                cluster_df_dict_basedon_hour_gap[cluster_name] = cluster_df_dict_basedon_hour_gap[cluster_name].drop([index_hour_greater_than_1[i+1]],0)
                
    for k,v in cluster_df_dict_basedon_hour_gap.items():
        print('Clusters: {}, Shape: {}'.format(k,v.shape))
      
    #Hyper-params    
    input_width = np.linspace(3,10,5,dtype=np.int64)
    latent_dim = 2
    input_feature_count = len(feature_column)
    epochs = [1]
    lr_1 = [0.001,0.01,0.1]
    lr_2 = [0.001,0.01,0.1]
    batch_size = [64,128]
    n_splits = [5]
    
    # tracking uri path
    experiment_name = 'USAD'
    tracking_uri_path = 'logs/' + 'mlruns/'
    mlflow.set_tracking_uri(tracking_uri_path)
    mlflow.set_experiment(experiment_name)
    
    
    space = {'input_width': hp.choice('input_width', input_width),
             'lr_1': hp.choice('lr_1', lr_1),
             'lr_2': hp.choice('lr_2', lr_2),
             'batch_size': hp.choice('batch_size', batch_size),
             'epochs': hp.choice('epochs', epochs),
             'n_splits': hp.choice('n_splits', n_splits)
        }
    
    # Optimization function 
    def f(space):
        
        input_width = space['input_width']
        lr_1 = space['lr_1']
        lr_2 = space['lr_2']
        batch_size = space['batch_size']
        epochs = space['epochs']       
        n_splits = space['n_splits']
        
        processed_time_series_data = get_time_series_window_data(cluster_df_dict_basedon_hour_gap, min_max_scaler, 
                                                                input_width)
        dataset = get_train_validate_dataset(processed_time_series_data, batch_size, n_splits)
        
        encoder_hidden_unit_list = get_encoder_hiddenlayer_neurons_list(input_feature_count, latent_dim)
        decoder_hidden_unit_list = get_decoder_hiddenlayer_neurons_list(input_feature_count, latent_dim)
        
        encoder_ = encoder(encoder_hidden_unit_list)
        decoder_1 = decoder(decoder_hidden_unit_list)
        decoder_2 = decoder(decoder_hidden_unit_list)
        
        mean_score, std_score = train(epochs, encoder_, decoder_1, decoder_2, dataset, lr_1,lr_2)
                    
        cv_val_loss_1 = mean_score[0]
        cv_val_loss_2 = mean_score[1]
                       
        val_loss_1_std = std_score[0]
        val_loss_2_std = std_score[1]
        
        consolidated_score = (np.abs(cv_val_loss_1) + np.abs(cv_val_loss_2))/2
        
        # Log model params and metric and save model
        with mlflow.start_run() as mlflow_run:
            for key in space.keys():
                mlflow.log_param(key, space[key])
            
            mlflow.log_metric('Mean_Loss1_CV_Score', cv_val_loss_1)
            mlflow.log_metric('Mean_Loss2_CV_Score', cv_val_loss_2)
            
            mlflow.log_metric('Std_Loss1_CV_Score', val_loss_1_std)
            mlflow.log_metric('Std_Loss2_CV_Score', val_loss_2_std)
            
            encoder_path = mlflow.get_artifact_uri() + '/' + 'Encoder'
            decoder_1_path = mlflow.get_artifact_uri() + '/' + 'Decoder_1'
            decoder_2_path = mlflow.get_artifact_uri() + '/' + 'Decoder_2'
            
            encoder_.save(encoder_path)
            decoder_1.save(decoder_1_path)
            decoder_2.save(decoder_2_path)            
        
        return {'loss': consolidated_score , 'status': STATUS_OK}
    
    # Hyper parameter search 
    trials = Trials()
    best = fmin(fn=f,
                space=space,
                algo = tpe.suggest,
                max_evals = 1,
                trials = trials,
                rstate = np.random.RandomState(1))
    