# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:35:07 2021


"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

class DataSampling:
    
    def __init__(self, dataframe, input_features, label):
        '''
        
        Parameters
        ----------
        dataframe : padas Dataframe
            Dataframe.
        input_features : list
            List of input features.
        label : str
            Labels.

        Returns
        -------
        None.

        '''
        self.df = dataframe
        self.input_features = input_features
        self.label = label       
        
    def stratified_sampling(self, random_state = 0, test_size = 0.3, shuffle = True):
        '''
        Stratified Sampling

        Parameters
        ----------
        random_state : int, optional
            Random State. The default is 0.
        test_size : float, optional
            Test size. The default is 0.3.
        shuffle : bool, optional
            Shuffle data. The default is True.

        Returns
        -------
        X_train : pandas dataframe
            Train data.
        X_test : pandas dataframe
            Test data.
        y_train : pandas dataframe
            train label.
        y_test : pandas dataframe
            test label.

        '''
        
        X = self.df[self.input_features]
        y = self.df[self.label]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = test_size,
                                                        random_state = random_state,
                                                        shuffle = shuffle,
                                                        stratify = y)
        return  X_train, X_test, y_train, y_test
    
    def stratified_kfold(self, n_splits = 5, shuffle = True):
        '''
        Stratified KFold

        Parameters
        ----------
        n_splits : int, optional
            Splits. The default is 5.
        shuffle : bool, optional
            Shuffle data. The default is True.

        Returns
        -------
        X : Pandas Dataframe
            Input Data.
        y : Pandas Dataframe
            Output Label.
        kfold : KFold Object
            DESCRIPTION.

        '''
        X = self.df[self.input_features]
        y = self.df[self.label]
        
        kfold = StratifiedKFold(n_splits = n_splits, shuffle = shuffle)
        
        return X, y, kfold
        
    
    