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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from hyperopt import Trials, STATUS_OK, tpe, hp, fmin

import mlflow

from pickle import dump
import utility




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
    
        
def encoder(hidden_unit_list,activations):
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
    
    for unit,activation in zip(hidden_unit_list,activations):
        model.add(tf.keras.layers.Dense(unit, activation = activation))
         
    return model

def decoder(hidden_unit_list, activations):
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
    
    for unit,activation in zip(hidden_unit_list,activations):
        model.add(tf.keras.layers.Dense(unit, activation=activation))
   
    return model


class Autoencoder(Model):
  def __init__(self, latent_dim, input_feature_count, encoder_activations, decoder_activations):
    super(Autoencoder, self).__init__()
    
    
    encoder_hidden_unit_list = get_encoder_hiddenlayer_neurons_list(input_feature_count, latent_dim)
    decoder_hidden_unit_list = get_decoder_hiddenlayer_neurons_list(input_feature_count, latent_dim)
    
    self.encoder = encoder(encoder_hidden_unit_list,encoder_activations)
    self.decoder = decoder(decoder_hidden_unit_list,decoder_activations )
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

@tf.function
def compute_loss(model,batch_data):
    y_hat = model(batch_data)
    loss = tf.reduce_mean((y_hat - batch_data)**2)
    return loss


@tf.function
def loss_gradient(model,input_batch, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model,input_batch )
    loss_grad = tape.gradient(loss, model.trainable_variables)
    
    return loss, loss_grad
    



if __name__ == "__main__":
    
    # Load Data
    data_v1 = pd.read_csv('../data/Imputed_Merged_Data_V_1.csv',index_col=[0])
    feature_column = ['VL1', 'VL2', 'VL3', 'IL1', 'IL2', 'IL3', 'VL12',
       'VL23', 'VL31', 'INUT', 'FRQ']
    time_column = ['DeviceTimeStamp']
    
    # convert to date time
    data_v1['DeviceTimeStamp'] = pd.to_datetime(data_v1['DeviceTimeStamp'])
    
    
    # Train-Test split
    train, test = utility.split_train_test_data(data_v1, split_ratio = 0.8)
    
    # Scale data
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit_transform(train[feature_column])    
    
    dump(min_max_scaler, open('dumps/min_max_scaler.pkl', 'wb'))
    
    
    # create clusters splitting index where hour greater than 1 hr
    index_hour_greater_than_1 = train[train.data_collection_frequency/np.timedelta64(1,'h') >= 1].index
    cluster_df_dict_basedon_hour_gap =  utility.get_cluster_df_basedon_gap(train,index_hour_greater_than_1)
      
    #Hyper-params    
    input_width = np.linspace(3,10,5,dtype=np.int64)
    latent_dim = 2
    input_feature_count = len(feature_column)
    epochs = [100]
    lr_1 = [0.001,0.005, 0.01, 0.1]    
    batch_size = [64,128]
    n_splits = [5]
    
    # encoder activations
    encoder_l1 = ['relu','sigmoid','tanh']
    encoder_l2 = ['relu','sigmoid','tanh']
    
    # decoder activations
    decoder_l1 = ['relu','sigmoid','tanh']
    decoder_l2 = ['relu','sigmoid','tanh']
    decoder_l3 = ['relu','sigmoid','tanh']
    
    # tracking uri path
    experiment_name = 'AutoEncoder: Iter_3'
    tracking_uri_path = 'logs/' + 'mlruns/'
    mlflow.set_tracking_uri(tracking_uri_path)
    mlflow.set_experiment(experiment_name)
    
    
    space = {'input_width': hp.choice('input_width', input_width),
             'lr_1': hp.choice('lr_1', lr_1),             
             'batch_size': hp.choice('batch_size', batch_size),
             'epochs': hp.choice('epochs', epochs),
             'n_splits': hp.choice('n_splits', n_splits),
             'encoder_l1': hp.choice('encoder_l1', encoder_l1),
             'encoder_l2': hp.choice('encoder_l2', encoder_l2),
             'decoder_l1': hp.choice('decoder_l1', decoder_l1),
             'decoder_l2': hp.choice('decoder_l2', decoder_l2),
             'decoder_l3': hp.choice('decoder_l3', decoder_l3),
        }
    
    
    # Optimization function 
    def f(space):
        
        input_width = space['input_width']
        lr_1 = space['lr_1']        
        batch_size = space['batch_size']
        epochs = space['epochs']       
        n_splits = space['n_splits']
        
        # encoder activations
        encoder_activations = [space['encoder_l1'], space['encoder_l2']]
        
        # decoder activations
        decoder_activations = [space['decoder_l1'], space['decoder_l2'],space['decoder_l3'] ]
        
        
        # time series data
        processed_time_series_data, timestamps_window = utility.get_time_series_window_data(cluster_df_dict_basedon_hour_gap, min_max_scaler, 
                                                                input_width, feature_column, time_column)
        dataset = utility.get_train_validate_dataset(processed_time_series_data, batch_size, n_splits)
        
        # Train
        train_cv_score = list()
        val_cv_score = list()
        for train_data, validate_data in dataset:            
            
            optimizer = tf.keras.optimizers.Adam(learning_rate = lr_1)
            autoencoder = Autoencoder(latent_dim, input_feature_count, encoder_activations, decoder_activations)
            
            # For early stopping
            patience = 10
            wait = 0
            best = 0

            for epoch in range(epochs):
                start = time.time()
                total_loss = 0                
                count = 0
                
                for input_batch in train_data.as_numpy_iterator():
                    if epoch == 0:
                        z = autoencoder(input_batch)
                        
                    loss, loss_grad = loss_gradient(autoencoder,input_batch, optimizer)
                    optimizer.apply_gradients(zip(loss_grad, autoencoder.trainable_variables))
                    total_loss += loss
                    count +=1
                
                total_loss = total_loss/count
                
                total_val_loss = 0
                count = 0
                for val_batch in validate_data.as_numpy_iterator():
                    val_loss  = compute_loss(autoencoder,val_batch )
                    total_val_loss += val_loss                    
                    count += 1
                total_val_loss = total_val_loss/count
                
                if (epoch+1)%5 == 0:
                    print('Epoch {}: ,train_loss = {}, val_loss = {},Time to run: {}'.format(epoch+1, 
                                                                                             total_loss, 
                                                                                             total_val_loss,
                                                                                             time.time() - start))
                # The early stopping strategy: stop the training if `val_loss` does not
                # decrease over a certain number of epochs.
                wait += 1
                if total_val_loss > best:
                  best = val_loss
                  wait = 0
                if wait >= patience:
                  break                     
            val_cv_score.append(total_val_loss)
            train_cv_score.append(total_loss)
        
        mean_train_score = np.mean(train_cv_score)
        std_train_score = np.std(train_cv_score)
        
        mean_cv_score = np.mean(val_cv_score)
        std_cv_score = np.std(val_cv_score)   
        with mlflow.start_run() as mlflow_run:
            for key in space.keys():
                mlflow.log_param(key, space[key])
            
            mlflow.log_metric('mean_cv_score', mean_cv_score)
            mlflow.log_metric('std_cv_score', std_cv_score)
            
            mlflow.log_metric('mean_train_score',mean_train_score )
            mlflow.log_metric('std_train_score',std_train_score )
            
            autoencoder_path = mlflow.get_artifact_uri() + '/' + 'Model'
            autoencoder.save(autoencoder_path)

        return {'loss': mean_cv_score , 'status': STATUS_OK}
    
    # Hyper parameter search 
    trials = Trials()
    best = fmin(fn=f,
                space=space,
                algo = tpe.suggest,
                max_evals = 2,
                trials = trials,
                rstate = np.random.RandomState(1))
    