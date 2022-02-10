# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 08:17:58 2022

@author: lokes
"""
import numpy as np
import pandas as pd
import datetime

from model_inference import ModelInference
import utility
    


input_width = 8
batch_size = 128
feature_column = ['VL1', 'VL2', 'VL3', 'IL1', 'IL2', 'IL3', 'VL12','VL23', 'VL31', 'INUT', 'FRQ']
time_column = 'DeviceTimeStamp'
run_id = 'logs/mlruns/1/9bfba4807e4046dd990c00219ab929a9/artifacts/Model'
scaler_path = 'dumps/min_max_scaler.pkl'
threshold = 0.14


# Load Data
data_v1 = pd.read_csv('../data/Imputed_Merged_Data_V_1.csv', index_col=[0])
data_v1[time_column] = pd.to_datetime(data_v1[time_column])


# Train-Test split
train, test = utility.split_train_test_data(data_v1, split_ratio = 0.8)

inference = ModelInference(input_width,batch_size,feature_column,time_column, run_id,scaler_path)

# aggregated_mse_loss, feature_mse_loss, feature_loss_dict  = inference.get_mse_loss(train, 'data_collection_frequency')


# model inference and time window
model_inference, time_window = inference.plot_latent_vector(train, 'data_collection_frequency',threshold)

# anomaly_points
anomaly_index = np.where(model_inference==1)[0]


# Time series plot
anomaly_seq = train[train.DeviceTimeStamp.astype('str').isin([str(pd.to_datetime(i)) for i in time_window[anomaly_index[1]]])]

