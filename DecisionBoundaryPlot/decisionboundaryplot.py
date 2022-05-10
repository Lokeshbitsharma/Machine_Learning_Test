# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:57:40 2022

@author: lokes
"""

import numpy as np
import pandas as pd
import configparser
import ast
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

from data_preparation.preprocess_data import LoadTransformData
from prediction_service.model_prediction import ModelPredictionService
from data_preparation.feature_meta_data import FeatureMetaData

from matplotlib.ticker import MaxNLocator # needed for integer only on axis
from matplotlib.lines import Line2D # for creating the custom legend
from constants import Constants


class DecisionBoundaryPlot:
    '''This class plots decision boundary and data points from high dimension to lower 2-Dimension '''
    
    def __init__(self, config_path, purpose_of_loan, optimal_threshold):
        '''
        Parameters
        ----------
        config_path : str
            configuration file path.
        purpose_of_loan : str
            Loan type. Example:'Business' .
        optimal_threshold : float
            Threshold for obtaining optimal decision boundary.

        Returns
        -------
        None.

        '''
        self.purpose_of_loan = purpose_of_loan        
        parser = configparser.ConfigParser()
        parser.read(config_path)        
        model_path = Constants.model_path   
        
        self.optimal_threshold = optimal_threshold
        self.prediction_service = ModelPredictionService(model_path, config_path)        
        self.feature_meta_data = FeatureMetaData() 
        
        self.dimensionality_reduction_model = tf.keras.models.Model(
            inputs = self.prediction_service.model.inputs,
            outputs = self.prediction_service.model.get_layer(index=-2).output,)
        
        self.out_model = self.__get_lastlayer_model()
        self.load_transform_data = LoadTransformData(config_path)  
        self.line_data = dict()
        
    def plot_decision_boundary(self,X,y, is_transformed = False, changed_threshold = None):
        '''
        This method is responsible for ploting high-dimension data as well as decision boundary into 2-D.
        Parameters
        ----------
        X : pandas data frame
            Input data.
        y : pandas series
            Output data.
        is_transformed : bool, optional
            True then data is already transformed, else transformation will be needed. The default is False.
        changed_threshold : list, optional
            List of threshold for shifted decision boundary.If None, optimal threshold value will be considered. The default is None.

        Returns
        -------
        None.

        '''        
        
        self.x0 , self.x1 = self.get_reduced_dimensions(X.copy(),is_transformed)        
        self.x0_axis_range, self.x1_axis_range = self.get_2D_axis_range(self.x0 , self.x1)
        
        self.xx0 , self.xx1 = np.meshgrid(self.x0_axis_range, self.x1_axis_range)
        self.xx = np.reshape(np.stack((self.xx0.ravel(),self.xx1.ravel()),axis=1),(-1,2))        
        
        
        # Prediction based on Optimal Threshold
        yy_hat = self.get_prediction_for_meshgrid(self.optimal_threshold)        
        yy_prob = self.get_prediction_prob_for_meshgrid()
        yy_size = self.get_grid_point_size(yy_prob)
        
        # Scale dot points in grid
        PROB_DOT_SCALE = 40 # modifier to scale the probability dots
        PROB_DOT_SCALE_POWER = 3 # exponential used to increase/decrease size of prob dots
        TRUE_DOT_SIZE = 50 # size of the true labels
                
        
        # make figure
        plt.style.use('seaborn-whitegrid') # set style because it looks nice
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10), dpi=150)
         
        #colormaps
        redish = '#d73027'
        blueish = '#4575b4'
        colormap = np.array([redish,blueish])
        
        # Original Plot
        ax.scatter(self.xx[:,0], self.xx[:,1], c = colormap[yy_hat].squeeze(), alpha=0.4, 
                    s = PROB_DOT_SCALE*yy_size**PROB_DOT_SCALE_POWER,
                    linewidths=0,)
       
        ax.contour(self.x0_axis_range, self.x1_axis_range, 
                    np.reshape(yy_hat,(self.xx0.shape[0],-1)), 
                    levels=1,
                    linewidths=1, 
                    colors=[colormap[0],colormap[1]])
    
        self.__update_line_data(self.optimal_threshold,ax.collections[-1])
        
        # Prediction based on changed threshold
        if changed_threshold is not None:
            for threshold in changed_threshold:
                if threshold != self.optimal_threshold:
                    yy_hat_changed_threshold = self.get_prediction_for_meshgrid(threshold) 
                    ax.contour(self.x0_axis_range, self.x1_axis_range, 
                                np.reshape(yy_hat_changed_threshold,(self.xx0.shape[0],-1)), 
                                levels=1,
                                linewidths=1,
                                colors=['#800000'],
                                linestyles='dashed') 
                    self.__update_line_data(threshold,ax.collections[-1])

        ax.scatter(self.x0, self.x1, c = colormap[y], s = TRUE_DOT_SIZE, zorder=3, linewidths=0.7, edgecolor='k')
        
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        ax.set_ylabel(r"$D_1$")
        ax.set_xlabel(r"$D_0$")
        ax.set_aspect(1)
        
        legend_class = []
        for loan_status, color in zip([Constants.approved, Constants.rejected], [blueish, redish]):
            legend_class.append(Line2D([0], [0], marker='o', label = loan_status, ls='None',
                                       markerfacecolor = color, markersize = np.sqrt(TRUE_DOT_SIZE), 
                                       markeredgecolor='k', markeredgewidth=0.7))

        legend1 = ax.legend(handles=legend_class, loc='lower left', 
                            bbox_to_anchor=(1.05, 0.35),
                            frameon=True, title='Loan Status')

        ax.add_artist(legend1)        
        ax.grid(False)
        
        legend_threshold = []
        for i in list(self.line_data.keys()):
            if self.optimal_threshold == i:
                legend_threshold.append(Line2D([0], [0],
                                               marker='o',
                                               label = i, 
                                               color = 'red',
                                               # ls ='None',
                                               # markerfacecolor = color,
                                               # markersize = np.sqrt(TRUE_DOT_SIZE), 
                                               # markeredgecolor='k',
                                               # linestyles='dashed',
                                               markeredgewidth=0.7))
            else:
                legend_threshold.append(Line2D([0], [0],
                                               marker='o',
                                               label = i, 
                                               # ls ='None',
                                               # markerfacecolor = color,
                                               # markersize = np.sqrt(TRUE_DOT_SIZE), 
                                               # markeredgecolor='k',                                               
                                               color = 'red',
                                               ls='dashed',
                                               markeredgewidth=0.7))
        
        legend2 = ax.legend(handles=legend_threshold, loc='upper left', 
                            bbox_to_anchor=(1.05, 0.35),
                            frameon=True, title='Thresholds')
        ax.add_artist(legend2)        
        ax.grid(False)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xticks(ax.get_xticks()[1:-1])
        ax.set_yticks(ax.get_yticks()[1:-1])
        # ax.set_aspect(1)
        if changed_threshold is not None: 
            plt.title( 'Decision Intelligence : ' + self.purpose_of_loan + '\n' +
                      'Threshold: ' + str(self.optimal_threshold) + '\n' + 
                      'Changed Threshold: ' + str(changed_threshold))
        else:
            plt.title( 'Decision Intelligence : ' + self.purpose_of_loan + '\n' +
                      'Threshold: ' + str(self.optimal_threshold) + '\n')
            
        
        # plt.show()
        
    def get_prediction_for_meshgrid(self,p):
        '''
        This method provides prediction for meshgrid based on set threshold p.
        Parameters
        ----------
        p : float
            Threshold value.

        Returns
        -------
        yy_hat : array
            Prediction.

        '''
        yy_hat = (self.out_model.predict(self.xx, steps =1) > p).astype("int32")
        return yy_hat
    
    def get_prediction_prob_for_meshgrid(self):
        '''
        This method returns prediction score for each points in mesh grid.
        '''
        yy_prob = self.out_model.predict(self.xx)
        return yy_prob
        
    def get_grid_point_size(self, yy_prob):
        '''
        This method returns size each point in the mesh grid based on probability score.

        Parameters
        ----------
        yy_prob : array
            Prediction probablity score.

        Returns
        -------
        yy_size : array
            Size based on probabilty score.

        '''
        yy_size = np.max(yy_prob, axis=1)         
        yy_size = np.array([i if i >= self.optimal_threshold else (1-i) for i in yy_size])
        return yy_size
        
    def get_reduced_dimensions(self,X, is_transformed = False):
        '''
        This method transforms high dimensional data to 2-D.

        Parameters
        ----------
        X : pandas data frame
            Input Data.
        is_transformed : bool, optional
            True then data is already transformed, else transformation will be needed. The default is False.

        Returns
        -------
        x0 : array
            Dimension-1 value.
        x1 : array
            Dimension-2 value.

        '''
        if is_transformed == False:
            X = self.load_transform_data.transform_input_data(X)
        bottleneck_values = self.dimensionality_reduction_model.predict(X)
        x0 = bottleneck_values[:,0]
        x1 = bottleneck_values[:,1]
        return x0,x1
    
    def get_2D_axis_range(self,x0, x1):
        '''
        This method returns 2-D axis range based on reduced dimension D1 and D2.

        Parameters
        ----------
        x0 : array
            Dimension-1.
        x1 : array
            Dimension-2.

        Returns
        -------
        x0_axis_range : array
            Dimension-1 axis range.
        x1_axis_range : array
        Dimension-2 axis range.

        '''
        PAD = .5 # how much to "pad" around the min/max values. helps in setting bounds of plot

        x0_min, x0_max = np.round(x0.min())-PAD, np.round(x0.max()+PAD)
        x1_min, x1_max = np.round(x1.min())-PAD, np.round(x1.max()+PAD)    
        
        H = .01 # mesh stepsize
        x0_axis_range = np.arange(x0_min,x0_max, H)
        x1_axis_range = np.arange(x1_min,x1_max, H)
        
        return x0_axis_range,x1_axis_range
        
    def __get_lastlayer_model(self):
        '''
        This method returns last layer output

        Returns
        -------
        out_model : model instance 
            Wrapper to fetch model last layer.

        '''
        out_layer_input = tf.keras.Input(shape=(2,))
        out_layer_output = self.prediction_service.model.layers[-1](out_layer_input)
        out_model = tf.keras.Model(out_layer_input, out_layer_output)   
        return out_model       
 
    def __update_line_data(self, threshold,ax_linecollection):        
        if type(ax_linecollection) == matplotlib.collections.LineCollection:
            self.line_data[threshold] = ax_linecollection.get_paths()[0].vertices
        
    def get_minmax_threshold(self):
        
        # make figure
        plt.style.use('seaborn-whitegrid') # set style because it looks nice
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,10), dpi=150)
         
        # Finding min and max threshold values
        thresh_range = np.linspace(0,1,100)
        thresh_range = [round(i,2) for i in thresh_range]
        applicable_threshold_values = []
        empty_thresh_values = []
        for threshold in thresh_range:
            yy_hat_changed_threshold = self.get_prediction_for_meshgrid(threshold) 
            ax.contour(self.x0_axis_range, self.x1_axis_range, 
                        np.reshape(yy_hat_changed_threshold,(self.xx0.shape[0],-1)), 
                        levels=1,
                        linewidths=1,
                        colors=['#800000'],
                        linestyles='dashed') 
            if len(ax.collections[-1].get_paths()) == 0:
                empty_thresh_values.append(threshold)
            else:
                applicable_threshold_values.append(threshold)
        plt.clf()
        return applicable_threshold_values[0],applicable_threshold_values[-1]
         