# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:53:52 2022

@author: lokes
"""

# -*- coding: utf-8 -*- Adversarial Result Generator

import numpy as np
import pandas as pd

import configparser

import numpy 


from prediction_service.model_prediction import ModelPredictionService
from data_preparation.feature_meta_data import FeatureMetaData


from constants import Constants
from AdversarialAI.zoo import ZOO
from data_preparation.preprocess_data import LoadTransformData
import tensorflow as tf
from tensorflow.io import gfile

from hyperopt import Trials, STATUS_OK, rand, tpe, hp, fmin

def validate_path(result_path):
    path_ = None
    for path in result_path.split('/'):
      if path_ is None:
          path_ = path
      else:
          path_ = path_ + '/' + path
      if not gfile.exists(path_):
          gfile.mkdir(path_)


purpose_of_loan = 'Business'
config_path = Constants.config_path
model_path = Constants.model_path
feature_meta_data = FeatureMetaData() 

#Load Data
parser = configparser.ConfigParser()
parser.read(config_path)
data_path = Constants.data_path
df = pd.read_csv(data_path)  
df.GoodCustomer = df.GoodCustomer.replace({-1:0})    
#df = df[df.PurposeOfLoan == purpose_of_loan]   

df = df[feature_meta_data.ALL_COLUMNS]
X = df[feature_meta_data.FEATURE_COLUMNS]
y = df[feature_meta_data.LABEL_COLUMN]


# Transform Data
load_transform = LoadTransformData(config_path)
x_transformed = load_transform.transform_input_data(X)

# convert to tensor
x_input = tf.convert_to_tensor(x_transformed.values,dtype = tf.float32)
y = tf.convert_to_tensor(y.values.reshape(-1,1),dtype = tf.float32)


def get_inverse_transform(x_input):
    df = pd.DataFrame(data = tf.reshape(x_input,[-1,len(feature_meta_data.FEATURE_COLUMNS)]).numpy(),
                      columns = feature_meta_data.FEATURE_COLUMNS)
    x = load_transform.inverse_transform_data(df)
    return x

def clip_bool_variables(input_x):
    bool_indexs = [feature_meta_data.FEATURE_COLUMNS.index(i) for i in feature_meta_data.BOOL_COLUMNS]
    data_points = list()
    for idx_ in range(input_x.shape[0]):
        data = input_x[idx_].numpy()
        for bool_idx in bool_indexs:                
            if data[bool_idx] <= 0.5:
                data[bool_idx] = 0.0
            else:
                data[bool_idx] = 1.0  
        data_points.append(data)
    
    cliped_variable = tf.convert_to_tensor(data_points,dtype = input_x.dtype)  

    return cliped_variable

# ###########################################################################################


from advesarial_samples.adversial_attack import AdversialAttack
from advesarial_samples.attacked_instance import AttackedInstance

box_max = x_transformed.max().values
box_min = x_transformed.min().values
prediction_service = ModelPredictionService(model_path, config_path)

list_attacked_instance = list()


config_path = Constants.config_path
model_path = Constants.model_path
decision_threshold = 0.49
max_iterations = 100
max_evals = 15

space = {'initial_const': hp.choice('initial_const',[0.0001,0.0005,0.001,0.005,0.01,0.05]),
          'confidence': hp.choice('confidence',[0.0,0.1]),
          'learning_rate': hp.choice('learning_rate',[0.001,0.01]),         
          }

for index in range(x_input.shape[0]):    
    input_data = tf.reshape(x_input[index],[1,-1])
    y_actual = tf.reshape(y[index],[-1,1])   
    y_actual_before_pertubation =  1 if prediction_service.model(input_data).numpy()[0][0] > decision_threshold else 0
    
    if int(y_actual.numpy()[0][0]) == y_actual_before_pertubation:
        ad_attack = AdversialAttack(space,
                        input_data,
                        model_path, config_path,
                        decision_threshold,                 
                        box_max,
                        box_min,
                        max_iterations,
                        max_evals)
    
        x_pertubation, loss = ad_attack.get_best_attack()
    
        y_pertub = 1 if prediction_service.model(x_pertubation).numpy()[0][0] > decision_threshold else 0
        
        if y_pertub != y_actual_before_pertubation:
            
            attack_instance = AttackedInstance()
            
            attack_instance.purtub_score = prediction_service.model(x_pertubation).numpy()[0][0]
            attack_instance.actual_score = prediction_service.model(input_data).numpy()[0][0]
            attack_instance.loss = loss
            attack_instance.actual_input = get_inverse_transform(input_data).T
            attack_instance.purtub_input = get_inverse_transform(x_pertubation).T 
            attack_instance.index = index
            list_attacked_instance.append(attack_instance)
            print("Adversial Sample Count : {}".format(len(list_attacked_instance)))

validate_path(Constants.adversarial_results_path)
actual_values_list = list()
features = feature_meta_data.FEATURE_COLUMNS
features.append('Prediction')
features.append('ApplicationID')
for i in range(len(list_attacked_instance)):
    values = list(list_attacked_instance[i].actual_input.T.values[0])
    values.append(list_attacked_instance[i].actual_score)
    values.append(list_attacked_instance[i].index)
    actual_values_list.append(values)
actual_datapoints = pd.DataFrame(columns=features, data=actual_values_list)
actual_datapoints['LoanStatus'] = np.where(actual_datapoints['Prediction'] > decision_threshold, 'Accepted', 'Rejected')
actual_datapoints.to_csv(Constants.adversarial_results_path + 'adversarial_result_actual_datapoints.csv', index = False) 

purturbed_values_list = list()
for i in range(len(list_attacked_instance)):
    values = list(list_attacked_instance[i].purtub_input.T.values[0])
    values.append(list_attacked_instance[i].purtub_score)
    values.append(list_attacked_instance[i].index)
    purturbed_values_list.append(values)
purturbed_datapoints = pd.DataFrame(columns=features, data=purturbed_values_list)
purturbed_datapoints['LoanStatus'] = np.where(purturbed_datapoints['Prediction'] > decision_threshold, 'Accepted', 'Rejected')
purturbed_datapoints.to_csv(Constants.adversarial_results_path + 'adversarial_result_purturbed_datapoints.csv', index = False) 

'''comparing actual and purturbed values'''
# if len(list_attacked_instance) > 0:
#     index_ = 0
#     comparision_df = pd.DataFrame(columns=['Actual','Pertubation'])
    
#     comparision_df['Actual'] = list_attacked_instance[index_].actual_input.T.values[0]
#     comparision_df['Pertubation'] = list_attacked_instance[index_].purtub_input.T.values[0]

#     comparision_df.index = feature_meta_data.FEATURE_COLUMNS
    
#     comparision_df.loc['Prediction',:] = [list_attacked_intance[index_].actual_score, 
#                                           list_attacked_instance[index_].purtub_score]
# else:
#     raise Exception('No Adverserial Sample Found.')

