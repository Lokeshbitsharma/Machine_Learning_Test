# -*- coding: utf-8 -*-


import os
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt


from Data_Preprocessing.TimeSeries_DataExtractor import TimeSeries_DataExtractor
from Data_Preprocessing.TimeSeries_WindowGenerator import WindowGenerator


class Model_Evaluation:
    
    def __init__(self,                             
                 label_dict,
                 config_params,
                 time_series_params,
                 task_path_dictionary,
                 retrieve_scaler=True):
        
        self.input_label = label_dict['input_label']
        self.output_label = label_dict['output_label']           
        self.index_label = label_dict['index_label']
        
        
        self.input_width = time_series_params.input_width
        self.label_width = time_series_params.label_width 
        self.shift = time_series_params.shift 
        self.shuffle = time_series_params.shuffle 
        
        self.data_extractor = TimeSeries_DataExtractor(task_path_dictionary,
                                                       label_dict,
                                                       config_params,
                                                       retrieve_scaler)
       
        
    def Get_Input_Output_Data(self,task_path_id):
        input_output_df = self.data_extractor.Get_TaskSpecific_InputOutputData(task_path_id)

        if any(True for output in self.output_label if output in self.input_label):
            input_output_label = self.input_label
        else:
            input_output_label = self.input_label + self.output_label
        
                    
        input_output_df = input_output_df[input_output_label]  
        
        window_generator = WindowGenerator( self.input_width,  self.label_width, self.shift,
                input_output_df, self.output_label, self.shuffle)
        
        X,y = window_generator.get_train_test_data()
        
        return X, y
  
                
    def Model_Point_to_Point_Based_Accuracy(self, y_pred,y_actual):    
        return np.array([pred/actual if (pred<actual) else (2 - (pred/actual)) 
                      for y_pred_,y_actual_ in zip(y_pred,y_actual)
                      for pred,actual in zip(y_pred_, y_actual_)]).reshape(-1,self.label_width)
    
    def Model_Point_Based_Accuracy(self,y_pred,y_actual):        
        return np.array([pred/actual if pred<actual else (2 - (pred/actual)) for pred,actual in zip(y_pred,y_actual)])
        
    
    
    def Get_Model_NonZero_Average_Prediction(self,y):        
        return self.__Get_NonZero_Average_From_MatrixDiagonalElement(y, self.label_width)
    
    
    
    def Plot(self, y_pred, y_actual,index= None, 
             x_label = 'Time', y_label='Revenue in million USD', title = 'Revenue Prediction'):
        
        if index is None:
            index = np.arange(0,len(y_pred),1)
            
        plt.plot(index, y_pred, color = 'blue', label='Prediction', marker = 'o')
        plt.plot(index, y_actual, color='green', label='Ground Truth', marker = '^')
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(rotation =45)
        plt.legend()
        plt.show()     
    
    
    def __Get_Diagonal_Elements(self, mat, n): 
        # principal = 0
        # secondary = 0
        principal = list()
        secondary = list()
        
        for i in range(0, n):  
            for j in range(0, n):  
      
                # Condition for principal diagonal 
                if (i == j):
                    principal.append(mat[i][j])
                    
      
                # Condition for secondary diagonal 
                if ((i + j) == (n - 1)): 
                    secondary.append(mat[i][j])
              
        return principal,secondary
    
        
    def __Get_NonZero_Average_From_MatrixDiagonalElement(self, input_array, label_width):
        
        mat = np.zeros(shape=(label_width,label_width)).tolist()
        dia_element_list = list()
        for i,pred in enumerate(input_array):    
            mat.append(pred.tolist())   
            mat.pop(0)
            p,s = self.__Get_Diagonal_Elements(mat,label_width)
            dia_element_list.append(s)
            if i == len(input_array) - 1:
                for x in range(label_width-1):
                    mat.append(np.zeros(label_width))
                    mat.pop(0)
                    p,s = self.__Get_Diagonal_Elements(mat,label_width)
                    dia_element_list.append(s)
                    
        dia_elements = np.array(dia_element_list)
        
        #return non-zero mean
        return np.true_divide(dia_elements.sum(1),(dia_elements!=0).sum(1))  
        
        
