# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:03:53 2021

@author: JY20144962
"""



import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import activations 


class Stacked_Model(keras.Model):
    def __init__(self,
                 seq_length=3,
                 n_features=3,                 
                 neurons_l1=64,
                 neurons_l2 = 64,                 
                 neurons_out = 1,                 
                 activationFn_1 = activations.relu,
                 activationFn_2 = activations.relu,
                 activationFn_3 = activations.relu):
            
            self.config = {'seq_length' : seq_length,
                'n_features' : n_features,
                'neurons_l1': neurons_l1,
                'neurons_l2': neurons_l2,                
                'neurons_out': neurons_out,                
                'activationFn_1': activationFn_1,
                'activationFn_2': activationFn_2,
                'activationFn_3': activationFn_3                
                }
        
            super(Stacked_Model,self).__init__()
            
            self.neurons_out = neurons_out
            self.hidden1 = keras.layers.LSTM(neurons_l1,return_sequences=True, 
                                             input_shape = (seq_length,n_features))
            self.hidden2 = keras.layers.LSTM(neurons_l2, return_sequences=False)
            self.out = keras.layers.Dense(neurons_out)
                        
            self.activation_l1 = activations.get(activationFn_1)
            self.activation_l2 = activations.get(activationFn_2)
            self.activation_l3 = activations.get(activationFn_3)
    
    def call(self, x):
        x = self.activation_l1(self.hidden1(x))        
        x = self.activation_l2(self.hidden2(x))
        x = self.activation_l3(self.out(x))
        
        x = tf.reshape(x,[-1,self.neurons_out ,1])
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

 