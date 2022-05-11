import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import activations 

class Custom_Model(keras.Model):
    
    def __init__(self,
                 input_shape=(1,1),
                 neurons_l1 = 40,
                 neurons_l2 = 40,
                 neurons_out = 1,
                 act_l1 = activations.relu,
                 act_l2 = activations.relu):        
       
    
        self.config = {'input_shape' : input_shape,
                'neurons_l1' : neurons_l1,
                'neurons_l2': neurons_l2,
                'neurons_out': neurons_out,
                'act_l1': act_l1,
                'act_l2': act_l2 }
        
        super(Custom_Model,self).__init__()
        
        self.hidden1 = keras.layers.Dense(neurons_l1, input_shape = input_shape)
        self.hidden2 = keras.layers.Dense(neurons_l2)
        self.out = keras.layers.Dense(neurons_out)
        self.activation_l1 = activations.get(act_l1)
        self.activation_l2 = activations.get(act_l2)
        
        
    def call(self, x):
        x = self.activation_l1(self.hidden1(x))
        x = self.activation_l2(self.hidden2(x))
        x = self.out(x)
        return x
    
    def get_config(self):        
        return self.config 
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
   
 