# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:49:14 2022

@author: lokes
"""

# Model Prediction Service
import numpy as np
import pandas as pd
import tensorflow as tf

from pickle import dump, load
import configparser
import ast
from data_preparation.preprocess_data import LoadTransformData
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, balanced_accuracy_score,fbeta_score,roc_curve, auc


class ModelPredictionService:
    
    def __init__(self, model_path, config_path):
        '''
        Model Prediction Service

        Parameters
        ----------
        model_path : str
            Model location path.
        config_path : str
            Configuration file path.

        Returns
        -------
        None.

        '''
        
        self.model = tf.keras.models.load_model(model_path)        
        self.load_transform_data = LoadTransformData(config_path)
        
        self.FEATURE_COLUMNS = self.load_transform_data.FEATURE_COLUMNS
        self.LABEL_COLUMN = self.load_transform_data.LABEL_COLUMN
        
    def get_predictions(self, X, p = 0.5, is_Transformed = False):
        '''
        Get predictions for input X

        Parameters
        ----------
        X : Pandas Dataframe
            Input data.
        p : float, optional
            probability threshold. The default is 0.5.

        Returns
        -------
        array
            Prediction for input X.

        '''
        if is_Transformed != True:
            X = self.load_transform_data.transform_input_data(X)
        X = tf.convert_to_tensor(X, dtype = tf.float64)
        return (self.model.predict(X, steps =1) > p).astype("int32")
    
    def get_prediction_proba(self, X, is_Transformed = False):
        if is_Transformed != True:
            X = self.load_transform_data.transform_input_data(X)
        X = tf.convert_to_tensor(X, dtype = tf.float64)
        return self.model.predict(X, steps=1)        
    
    def classification_report(self, X_actual, y_actual, p =0.5, is_Transformed = False):
        '''
        Classification Report

        Parameters
        ----------
        X_actual : Pandas Dataframe
            Input data.
        y_actual : TYPE
            Actual Output.
        p : float, optional
            probability threshold. The default is 0.5.
        Returns
        -------
        Text report
            
        '''                   
        y_pred = self.get_predictions(X_actual,p,is_Transformed)
        return classification_report(y_actual, y_pred)
    
    def confusion_matrix(self, X_actual, y_actual, p =0.5, is_Transformed = False):
        '''
        Confusion matrix

        Parameters
        ----------
        X_actual : Pandas Dataframe
            Input data.
        y_actual : TYPE
            Actual Output.
        p : float, optional
            probability threshold. The default is 0.5.
        Returns
        -------
        Confusion Matrix report
            
        '''
                
        y_pred = self.get_predictions(X_actual, p, is_Transformed)  
        return confusion_matrix(y_actual, y_pred)
        
    def balanced_accuracy(self,X_actual, y_actual, p=0.5, is_Transformed = False):    
        '''        
        Balanced Accuracy

        Parameters
        ----------
        X_actual : Pandas Dataframe
            Input data.
        y_actual : TYPE
            Actual Output.
        p : float, optional
            probability threshold. The default is 0.5.
        Returns
        -------
        Balanced Accuracy

        '''
       
        y_pred = self.get_predictions(X_actual,p,is_Transformed)
        return balanced_accuracy_score(y_actual, y_pred)
    
    def accuracy(self,X_actual, y_actual,  p=0.5, is_Transformed = False):   
        '''
        Accuracy

        Parameters
        ----------
        X_actual : Pandas Dataframe
            Input data.
        y_actual : TYPE
            Actual Output.
        p : float, optional
            probability threshold. The default is 0.5.
        Returns
        -------
        Accuracy

        '''
            
        y_pred = self.get_predictions(X_actual,p,is_Transformed)
        return accuracy_score(y_actual, y_pred)
    
    
    def F_score(self,X_actual, y_actual,beta = 1,  p=0.5, is_Transformed = False):  
        '''
        F-Score

        Parameters
        ----------
        X_actual :  Pandas Dataframe
            Input data.
        y_actual :  Pandas Dataframe
            Input data.
        beta : TYPE, optional
            Beta factor. The default is 1.
        p : float, optional
            probability threshold. The default is 0.5.

        Returns
        -------
        F-Score

        '''
       
        y_pred = self.get_predictions(X_actual,p,is_Transformed)        
        return fbeta_score(y_actual, y_pred,average='weighted',beta=beta)
    
    def AUC_Score(self,X_actual, y_actual,  p=0.5, is_Transformed = False):
        '''
        AUC Score

        Parameters
        ----------
        X_actual : Pandas Dataframe
            Input data.
        y_actual : TYPE
            Actual Output.
        p : float, optional
            probability threshold. The default is 0.5.
        Returns
        -------
        AUC Score

        '''
       
        y_pred = self.get_predictions(X_actual,p,is_Transformed) 
        fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
        return auc(fpr, tpr)
    
    def get_optimal_threshold_from_dataframe(self,df,beta = 1): 
        
        prediction_dict = dict()
        for i in np.linspace(0.1,0.9, 80):
            X_actual = df[self.FEATURE_COLUMNS]
            y_actual = df[self.LABEL_COLUMN]
            p = round(i,2)
            acc = self.F_score(X_actual, y_actual,beta, p)
            prediction_dict[p] = acc
        return max(prediction_dict, key= lambda x: prediction_dict[x])
    
    def get_optimal_threshold(self,X_actual, y_actual,beta = 1, is_Transformed = False):
        prediction_dict = dict()
        
        if is_Transformed == False:
            X_actual = self.load_transform_data.transform_input_data(X_actual)
        
        for i in np.linspace(0.1,0.9, 80):          
            p = round(i,2)
            acc = self.F_score(X_actual, y_actual,beta, p, is_Transformed = True)
            prediction_dict[p] = acc
        return max(prediction_dict, key= lambda x: prediction_dict[x])
        
   