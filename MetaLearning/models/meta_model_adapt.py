# -*- coding: utf-8 -*-
"""

"""

import tensorflow as tf
import tensorflow.keras.activations as activations


class Meta_Model_Adapt(tf.keras.Model):
    
    def __init__(self,
                 meta_model,
                  input_shape,
                  neurons_l1 = 50,
                  neurons_out = 1,
                  activation_l1 = 'relu',
                  activation_l2 = 'relu'):
        
        self.config = {
                'meta_model': meta_model,
                'input_shape': input_shape,
                'neurons_l1': neurons_l1,
                'neurons_out': neurons_out,
                'activation_l1': activation_l1,
                'activation_l2': activation_l2               
                }
        
        super(Meta_Model_Adapt,self).__init__()       
        
        
        self.meta_model = meta_model
        self.hidden_1 = tf.keras.layers.LSTM(neurons_l1, return_sequences=True,
                                                        input_shape = input_shape)
        self.hidden_2 = tf.keras.layers.Dense(neurons_out)
        self.activation_l1 = activations.get(activation_l1)
        self.activation_l2 = activations.get(activation_l2)
        
        
    def call(self, x):
        x = self.meta_model(x)
        x = self.activation_l1(self.hidden_1(x))
        x = self.activation_l2(self.hidden_2(x))
        
        return x
    
    
    def get_config(self):        
        return self.config 
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)