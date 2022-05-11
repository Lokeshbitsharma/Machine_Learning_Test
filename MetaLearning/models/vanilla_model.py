# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:03:53 2021

@author: JY20144962
"""

import tensorflow.keras as keras
from tensorflow.keras import activations 


class Vanilla_Model(keras.Model):
    def __init__(self,
                 seq_length=2,
                 n_features=2,                 
                 neurons_l1=128,
                 neurons_out = 3,
                 dropout_l1=0.1,
                 activationFn_1 = 'relu',
                 isDropout = True):
            
            self.config = {'seq_length' : seq_length,
                'n_features' : n_features,
                'neurons_l1': neurons_l1,
                'neurons_out': neurons_out,
                'dropout_l1': dropout_l1,
                'activationFn_1': activationFn_1,
                'isDropout' : isDropout
                }
        
            super(Vanilla_Model,self).__init__()
            
            
            self.hidden1 = keras.layers.LSTM(neurons_l1,return_sequences= False, 
                                             input_shape = (seq_length,n_features))
            self.out = keras.layers.Dense(neurons_out)
            
            self.isDropout = isDropout
            self.activation_l1 = activations.get(activationFn_1)
            self.dropout_l1 =  keras.layers.Dropout(dropout_l1)
            
            
    
    def call(self, x):
        x = self.activation_l1(self.hidden1(x))
        if self.isDropout == True:
            x = self.dropout_l1(x)
        x = self.out(x)
        return x
    
       
    def get_config(self):        
        return self.config 
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
###To Test the Model 
# x = np.array([[[1.0,2.0,3.0],
#                [4.0,5.0,6.0],
#                [7.0,8.0,9.0]]])

# x = x.reshape(1,3,3)
# y = np.array([1.0,2.0,3.0])
# model = Vanilla_Model()

# def loss_function(pred_y, y):
#         #return tf.keras.backend.mean(tf.keras.losses.MSE(y, pred_y))
#         return tf.reduce_mean(tf.keras.losses.MSE(y, pred_y)) 

# def compute_loss(model,x,y):
#         model_output = model(x)
#         loss = loss_function(y, model_output)
#         return loss, model_output


# def train_Model(x,y):
    
#     for i in range(500):
#         with tf.GradientTape() as train_tape:
#                     train_loss, y_hat_train = compute_loss(model,x,y)                  
#                     task_gradients = train_tape.gradient(train_loss,model.trainable_variables)
#         k = 0
#         for j in range(len(model.trainable_variables)):
#             model.trainable_variables[j].assign(tf.subtract(model.trainable_variables[j], 
#                                                                              tf.multiply(0.01, task_gradients[j])))
#             k += 2
#     return model
        
# model = train_Model(x,y)
# print(model(x))

