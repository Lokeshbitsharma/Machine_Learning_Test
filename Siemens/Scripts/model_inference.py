# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 06:59:08 2022

@author: lokes
"""

import numpy as np
import pandas as pd
import time
from datetime import datetime


import tensorflow as tf
import utility

from pickle import load
from matplotlib import pyplot as plt
import seaborn as sns

class ModelInference:
    
    def __init__(self,input_width,batch_size,feature_column, time_column, run_id,scaler_path):
        
        self.input_width = input_width
        self.batch_size = batch_size
        self.time_column = time_column
        self.run_id = run_id
        self.feature_column = feature_column
        
        self.scaler = load(open(scaler_path, 'rb'))
        self.model = tf.keras.models.load_model(run_id)
        
    def get_mse_loss(self, data, data_collection_frequency):
        
        # time series processed data
        data, time_window = self.__get_timeseries_window_data(data, data_collection_frequency)        
        aggregated_mse_loss, feature_mse_loss = self.__get_mse_loss_from_processed_data(data)
        
        # MSE loss plot feature specific
        feature_loss_dict = dict()
        for i,feature in enumerate(self.feature_column):
            feature_loss_dict[feature] = [losses[i] for losses in feature_mse_loss]
            
        return aggregated_mse_loss, feature_mse_loss, feature_loss_dict 
    
    def get_model_inference(self, mse_loss, loss_threshold):        
        anomaly_result = np.array([1 if i > loss_threshold else 0 for i in np.array(mse_loss)])        
        return anomaly_result
        
    def plot_mse_loss(self,feature_loss_dict, feature_name):        
        plt.hist(feature_loss_dict[feature_name], bins=50)
        plt.xlabel("Train MSE loss: {}".format(feature_name))
        plt.ylabel("No of samples")
        plt.show()
        
    def plot_latent_vector(self,data, data_collection_frequency, threshold = 0.1):
        
        # time series processed data
        time_series_data, time_window = self.__get_timeseries_window_data(data, data_collection_frequency)
        
        latent_vector = self.model.encoder(time_series_data)
        
        x = list()
        y = list()
        for vector in latent_vector:
            x.append(np.mean([i[0] for i in vector.numpy()]))
            y.append(np.mean([i[1] for i in vector.numpy()]))
        
        aggregated_mse_loss, feature_mse_loss = self.__get_mse_loss_from_processed_data(time_series_data)
        
        model_inference = self.get_model_inference(aggregated_mse_loss, threshold)        
        
        results_data = pd.DataFrame(columns = ['D1','D2','Anomaly'])
        results_data.D1 = x
        results_data.D2 = y
        results_data.Anomaly = model_inference
        
        sns.scatterplot(data= results_data, 
                        x = 'D1',
                        y = 'D2',
                        hue = 'Anomaly',
                        palette="deep")
        
        plt.title("Latent vector representation: Threshold: {}".format(threshold))
        plt.ylabel("D2")
        plt.xlabel("D1")
        plt.show()      
        
        return model_inference, time_window
    
    def plot_feature_mse_basedon_index(self,mse_loss,feature_mse_loss, anomaly_index, time_window):
        print('Anomaly MSE : {}'.format(mse_loss[anomaly_index]))
        sns.barplot(x=self.feature_column, y=feature_mse_loss[anomaly_index])
        plt.xlabel('Features')
        plt.ylabel('MSE')
        plt.title('Feature Vs MSE: \n Start Time: {} \n End Time: {} '.format(time_window[anomaly_index][0],time_window[anomaly_index][-1]))
        plt.show()
        
    def plot_timeseries_sequence(self, data, time_window, anomaly_index):
        anomaly_data = data[data[self.time_column].astype('str').isin([str(pd.to_datetime(i)) for i in time_window[anomaly_index]])]
        plot_features = anomaly_data[self.feature_column]
        plot_features.index = anomaly_data[self.time_column]
        _ = plot_features.plot(subplots=True)  
        
                
    
    def __get_timeseries_window_data(self, data, data_collection_frequency):
        index_hour_greater_than_1 = data[data[data_collection_frequency]/np.timedelta64(1,'h') >= 1].index
        cluster_df_dict_basedon_hour_gap =  utility.get_cluster_df_basedon_gap(data,index_hour_greater_than_1)
        
        
        # time series processed data
        data, time_window = utility.get_time_series_window_data(cluster_df_dict_basedon_hour_gap, self.scaler, 
                                                                self.input_width,
                                                                self.feature_column,
                                                                self.time_column)
        return data, time_window
        
    def __get_mse_loss_from_processed_data(self, data):
        data_predict = self.model.predict(data)
        feature_mse_loss = np.mean(np.abs(data_predict - data), axis=1)
        aggregated_mse_loss = [np.mean(i) for i in feature_mse_loss]
        return aggregated_mse_loss, feature_mse_loss
        
    
        
        
        
