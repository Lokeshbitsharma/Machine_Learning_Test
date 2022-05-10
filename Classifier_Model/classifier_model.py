# classifier Model

import tensorflow as tf


class ClassifierModel:
    ''' This class is used for provide instance of compiled model based on defined architectures.'''
    
    def __init__(self):
        pass
    
    def make_model_with_1_hidden_layer(self,input_shape, 
                   num_classes,
                   loss_func,
                   metrics,
                   n_layer1,
                   act_layer1,
                   learning_rate = 1e-3,
                   output_bias=None):
        '''
        Classifier model with one hidden layer.

        Parameters
        ----------
        input_shape : int
            Number of input features.
        num_classes : int
            Number of clasess in target variable.
        loss_func : str
            Loss funtion used for training.
        metrics : str
            Metrics to be evaluated by the model during training and testing.
        n_layer1 : int
            Number of neurons in layer 1.
        act_layer1 : str
            Activation function for layer 1.
        learning_rate : float, optional
            Learning rate used during training. The default is 1e-3.
        output_bias : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        model : keras.model
            DESCRIPTION.

        '''
        
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
            
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(n_layer1, activation = act_layer1,
              input_shape=(input_shape,)),
          # tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Dense(num_classes,
                                activation='sigmoid',
                                bias_initializer = output_bias)])
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                    loss= loss_func,
                    metrics = metrics)
        return model
    
    
    def make_model_with_2_hidden_layer(self,input_shape, 
                   num_classes,
                   loss_func,
                   metrics,
                   n_layer1,
                   act_layer1,
                   n_layer2,
                   act_layer2,
                   learning_rate = 1e-3,
                   output_bias=None):
        '''
        Classifier Model with 2 hidden layers.

        Parameters
        ----------
        input_shape : int
            Number of input features.
        num_classes : int
            Number of clasess in target variable.
        loss_func : str
            Loss funtion used for training.
        metrics : str
            Metrics to be evaluated by the model during training and testing.
        n_layer1 : int
            Number of neurons in layer 1.
        act_layer1 : str
            Activation function for layer 1.
        n_layer2 : int
            Number of neurons in layer 2.
        act_layer2 : str
            Activation function for layer 2.
        learning_rate : float, optional
            Learning rate used during training. The default is 1e-3.
        output_bias : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        model : keras.model
            DESCRIPTION.

        '''
        
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
            
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(n_layer1, activation = act_layer1,
              input_shape=(input_shape,)),
          # tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Dense(n_layer2, activation = act_layer2),
          tf.keras.layers.Dense(num_classes,
                                activation='sigmoid',
                                bias_initializer = output_bias)])
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                    loss= loss_func,
                    metrics = metrics)
        return model
    
    def make_model_with_3_hidden_layer(self,input_shape, 
                   num_classes,
                   loss_func,
                   metrics,
                   n_layer1,
                   act_layer1,
                   n_layer2,
                   act_layer2,
                   n_layer3,
                   act_layer3,
                   learning_rate = 1e-3,
                   output_bias=None):
        '''
        Classifier Model with 3 hidden layers.

        Parameters
        ----------
        input_shape : int
            Number of input features.
        num_classes : int
            Number of clasess in target variable.
        loss_func : str
            Loss funtion used for training.
        metrics : str
            Metrics to be evaluated by the model during training and testing.
        n_layer1 : int
            Number of neurons in layer 1.
        act_layer1 : str
            Activation function for layer 1.
        n_layer2 : int
            Number of neurons in layer 2.
        act_layer2 : str
            Activation function for layer 2.
        n_layer3 : int
            Number of neurons in layer 3.
        act_layer3 : str
            Activation function for layer 3.
        learning_rate : float, optional
            Learning rate used during training. The default is 1e-3.
        output_bias : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        model : keras.model
            DESCRIPTION.

        '''
        
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
            
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(n_layer1, activation = act_layer1,
              input_shape=(input_shape,)),
          # tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Dense(n_layer2, activation = act_layer2),
          tf.keras.layers.Dense(n_layer3, activation = act_layer3),          
          tf.keras.layers.Dense(num_classes,
                                activation='sigmoid',
                                bias_initializer = output_bias)])
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
                    loss= loss_func,
                    metrics = metrics
                    )
        return model
    
    def make_custom_model_with_3_hidden_layer(self,input_shape, 
                   num_classes,  
                   n_layer1,
                   act_layer1,
                   n_layer2,
                   act_layer2,
                   n_layer3,
                   act_layer3,
                   learning_rate = 1e-3,
                   output_bias=None):
        '''
        Classifier Model with 3 hidden layers  and custom loss funtion.

        Parameters
        ----------
        input_shape : int
            Number of input features.
        num_classes : int
            Number of clasess in target variable.      
        n_layer1 : int
            Number of neurons in layer 1.
        act_layer1 : str
            Activation function for layer 1.
        n_layer2 : int
            Number of neurons in layer 2.
        act_layer2 : str
            Activation function for layer 2.
        n_layer3 : int
            Number of neurons in layer 3.
        act_layer3 : str
            Activation function for layer 3.
        learning_rate : float, optional
            Learning rate used during training. The default is 1e-3.
        output_bias : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        model : keras.model
            DESCRIPTION.

        '''
        
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
            
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(n_layer1, activation = act_layer1,
              input_shape=(input_shape,)),
          # tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Dense(n_layer2, activation = act_layer2),
          tf.keras.layers.Dense(n_layer3, activation = act_layer3),          
          tf.keras.layers.Dense(num_classes,
                                activation='sigmoid',
                                bias_initializer = output_bias)])
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate))
        return model
    
    def make_custom_model_with_4_hidden_layer(self,input_shape, 
                   num_classes,  
                   n_layer1,
                   act_layer1,
                   n_layer2,
                   act_layer2,
                   n_layer3,
                   act_layer3,
                   n_layer4,
                   act_layer4,
                   learning_rate = 1e-3,
                   output_bias=None):
        '''
        Classifier Model with 4 hidden layers and custom loss funtion.
        
        Parameters
        ----------
        input_shape : int
            Number of input features.
        num_classes : int
            Number of clasess in target variable.      
        n_layer1 : int
            Number of neurons in layer 1.
        act_layer1 : str
            Activation function for layer 1.
        n_layer2 : int
            Number of neurons in layer 2.
        act_layer2 : str
            Activation function for layer 2.
        n_layer3 : int
            Number of neurons in layer 3.
        act_layer3 : str
            Activation function for layer 3.
        n_layer4 : int
        Number of neurons in layer 4.
        act_layer4 : str
        Activation function for layer 4.
        learning_rate : float, optional
            Learning rate used during training. The default is 1e-3.
        output_bias : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        model : keras.model
            DESCRIPTION.

        '''
        
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
            
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(n_layer1, activation = act_layer1,
              input_shape=(input_shape,)),
          # tf.keras.layers.Dropout(0.5),
          tf.keras.layers.Dense(n_layer2, activation = act_layer2),
          tf.keras.layers.Dense(n_layer3, activation = act_layer3),  
          tf.keras.layers.Dense(n_layer4, activation = act_layer4),
          tf.keras.layers.Dense(num_classes,
                                activation='sigmoid',
                                bias_initializer = output_bias)])
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate))
        return model