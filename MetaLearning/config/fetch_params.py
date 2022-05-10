# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:29:41 2022

@author: lokes
"""

import numpy as np
import gin

from data_preprocessing.custom_preprocessing import FeatureScalerType

@gin.configurable
class FetchParams:
    
    @gin.configurable
    def get_metadata_path(self, metadata_path):
        return metadata_path
    
    @gin.configurable
    def get_mlflow_uri_path(self, uri_path):
        return uri_path
    
    @gin.configurable
    def get_n_samples_k(self,n_samples_k):
        return n_samples_k
    
    @gin.configurable
    def get_n_test(self,n_test):
        return n_test
    
    @gin.configurable
    def get_epochs(self, epochs):
        return epochs
    
    @gin.configurable
    def get_alpha(self, alpha):
        return alpha
    
    @gin.configurable
    def get_beta(self, beta):
        return beta
    
    @gin.configurable
    def get_input_shape(self, shape):
        return shape
    
    @gin.configurable
    def get_neurons_l1_range(self, low, high, steps):
        return np.linspace(low, high, steps, dtype=np.int)
    
    @gin.configurable
    def get_neurons_l2_range(self, low, high, steps):
        return np.linspace(low, high, steps, dtype=np.int)
        
    @gin.configurable
    def get_neurons_out(self, neurons_out):
        return neurons_out
    
    @gin.configurable
    def get_activation_l1(self, activation_l1):
        return activation_l1
    
    @gin.configurable
    def get_activation_l2(self, activation_l2):
        return activation_l2
    
    @gin.configurable
    def get_input_label(self, input_label):
        return input_label
    
    @gin.configurable
    def get_output_label(self, output_label):
        return output_label
    
    @gin.configurable
    def get_time_index_label(self, index_label):
        return index_label
        
    @gin.configurable
    def get_scaler_feature_dict(self,scaler_list,feature_list):
        scaler_feature_dict = dict()
        
        i =0 
        for scaler, feature in zip(scaler_list, feature_list):
            scaler_enum = self.get_scaler_enum(scaler)
            scaler_feature_dict[scaler_enum + '_' + str(i)] = feature
            i += 1
        return scaler_feature_dict
    
    @gin.configurable
    def get_scaler_storage_path(self, scaler_storage_path):
        return scaler_storage_path
    
    def get_scaler_enum(self, scaler):
        if scaler == FeatureScalerType.standard_scaler.name:
            return FeatureScalerType.standard_scaler
        elif scaler == FeatureScalerType.minmax_scaler:
            return FeatureScalerType.minmax_scaler
        else:
            raise ValueError('Invalid FeatureScalerType {}'.format(scaler))
        