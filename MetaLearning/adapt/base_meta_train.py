# -*- coding: utf-8 -*-
"""
Base_model_train.py
"""
import tensorflow as tf
from abc import ABC,abstractmethod
from sklearn.model_selection import train_test_split
import tensorflow.keras.optimizers as optimizers

class Base_Model_Train(ABC):
    
    def __init__(self,model,
                 lossfunction,
                 epochs,
                 optimizer,
                 learning_rate,
                 test_size=0.30):
        
        self.__model = model
        self.loss_function = lossfunction
        self.epochs = epochs 
        
        self.optimizer = optimizers.get(optimizer)
        self.optimizer._hyper['learning_rate'] = learning_rate
        self.test_size = test_size
        
        
    def train_with_sampling(self,X, y):
        
        train_loss_list = list()
        test_loss_list = list()
        
        model = self.__model
        
        for i in range(self.epochs): 
            
            X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30)
            
            train_loss, model = self.__train_step(model,X_train, Y_train)
            
            y_pred = model(X_test)            
            test_loss = self.loss_function(y_pred,Y_test)   
            
            train_loss_list.append(train_loss.numpy())
            test_loss_list.append(test_loss.numpy())
            
            if (i+1) % 10 == 0:
                print("Epoch= {:d} Train loss={:.2f} Test loss={:.2f}".format(i+1,train_loss,test_loss))
            
        return train_loss_list, test_loss_list, model
    
    def train(self,X_train, X_test, Y_train, Y_test):
        train_loss_list = list()
        test_loss_list = list()
        
        model = self.__model
        
        for i in range(self.epochs):
            train_loss, model = self.__train_step(model,X_train, Y_train)
            y_pred = model(X_test)            
            test_loss = self.loss_function(y_pred,Y_test)   
            
            train_loss_list.append(train_loss.numpy())
            test_loss_list.append(test_loss.numpy())
            
            if (i+1) % 10 == 0:
                print("Epoch= {:d} Train loss={:.2f} Test loss={:.2f}".format(i+1,train_loss,test_loss))
            
        return train_loss_list, test_loss_list, model
        
        
        
    def __train_step(self,model, X, y):  
        with tf.GradientTape() as tape:
            pred_y = model(X)
            loss = self.loss_function(pred_y, y)
        # variables = tape.watched_variables()
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, model
    
        