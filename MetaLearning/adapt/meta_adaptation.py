# -*- coding: utf-8 -*-
"""
Meta Adapt
"""
import numpy as np
import pandas as pd


import tensorflow as tf
from Test_Generator import SinusoidGenerator
from sklearn.model_selection import train_test_split

from Adapt.Base_Model_Train import Base_Model_Train


class Meta_Adaptation(Base_Model_Train):
    
    def __init__(self,
                 loss_function,
                 model_path, 
                 custom_model_objects,
                 epochs,
                 optimizer,
                 learning_rate,
                 test_size=0.30):
        
        model = self.get_meta_trained_model(model_path,custom_model_objects)
        
        super().__init__(model, loss_function, epochs, optimizer, learning_rate, test_size)
        
    def get_meta_trained_model(self,model_path,custom_model_objects):        
        '''
            This method returns the meta trained model
        '''
        meta_trained_model = tf.keras.models.load_model(model_path, 
                                   custom_objects= custom_model_objects)
        return meta_trained_model     
