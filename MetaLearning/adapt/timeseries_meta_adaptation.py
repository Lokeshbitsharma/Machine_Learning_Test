# -*- coding: utf-8 -*-
"""
Timeseries Meta Adapt
"""

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
from sklearn.model_selection import train_test_split
from Models.Meta_Model_Adapt import Meta_Model_Adapt
from Adapt.Base_Model_Train import Base_Model_Train



class TimeSeries_Meta_Adaptation(Base_Model_Train):
    
    def __init__(self,
                 meta_model_path,
                 meta_model_custom_objects,
                 unfreeze_number_of_layers,                
                 meta_adapt_params_dict,
                 loss_function,
                 optimizer,
                 learning_rate,
                 epochs,
                 test_size=0.30):
        
        self.__meta_model = self.__Get_Meta_Model(meta_model_path, 
                                                meta_model_custom_objects, 
                                                unfreeze_number_of_layers)
        
        
        input_shape = meta_adapt_params_dict['input_shape']
        neurons_l1 = meta_adapt_params_dict['neurons_l1']
        neurons_out = meta_adapt_params_dict['neurons_out']
        activation_l1 = meta_adapt_params_dict['activation_l1']
        activation_l2 = meta_adapt_params_dict['activation_l2']
        
        
        self.meta_model_adapt = Meta_Model_Adapt(self.__meta_model,
                                                  input_shape,
                                                  neurons_l1,
                                                  neurons_out,
                                                  activation_l1,
                                                  activation_l2)
        
        super().__init__(self.meta_model_adapt,loss_function,epochs,optimizer,learning_rate,test_size)
    
    
    def __Get_Meta_Model(self, 
                       meta_model_path,
                       custom_objects,
                       unfreeze_number_of_layers):
        
        meta_Model =  tf.keras.models.load_model(meta_model_path, 
                                    custom_objects = custom_objects)
        
        for i in range(len(meta_Model.layers)):
            if i < len(meta_Model.layers) - unfreeze_number_of_layers:
                meta_Model.layers[i].trainable = False        
        
        return meta_Model
        
        
        
    
    