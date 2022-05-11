# -*- coding: utf-8 -*-
"""

"""

from enum import Enum
from Models.Model import Custom_Model
from Models.Vanilla_Model import Vanilla_Model
from Models.Stacked_Model import Stacked_Model


# Enum class to indicate the type of model to be returned from factory
class Model_Name(Enum):
    
    Custom_Model = 1,
    Vanilla_LSTM_Model = 2,
    Stacked_LSTM_Model = 3
    


class Model_Factory:
        
    def __init__(self,model_name):
        '''
        This class is used for creatng a new instance of model based on the model enum type.

        Parameters
        ----------
        model_name : enum
            Model_Name enum.

        Raises
        ------
        ValueError
            In case the Model_Name is invalid.

        Returns
        -------
        None.

        '''
        if model_name not in Model_Name._value2member_map_.values():
            raise ValueError('Invalid Model name name passed as an argument.')
            
        self.model_name = model_name
        
    def Get_Model(self, model_params_dict):
        '''
        This method returns model model instance based on model name and model params passed.

        Parameters
        ----------
        model_params_dict : dict
            model params.

        Returns
        -------
        Instance of the created model            

        '''
        
        if self.model_name.name == Model_Name.Custom_Model.name: 
            return self.__Get_Custom_Model(model_params_dict)
        
        elif self.model_name.name == Model_Name.Vanilla_LSTM_Model.name:
            return self.__Get_Vanilla_LSTM_Model(model_params_dict)
        
        elif self.model_name.name == Model_Name.Stacked_LSTM_Model.name:
            return self.__Get_Stacked_LSTM_Model(model_params_dict)
        else:
            raise ValueError("Incorrect Model Name")
            
            
            
    def __Get_Vanilla_LSTM_Model(self,model_params_dict):
        
        args_list = list(['input_width','n_features','neurons_l1','neurons_out',
                              'dropout_l1','act_l1','isDropout'])
        
        if all(key in args_list for key in model_params_dict.keys()) == False:
            raise ValueError('Missing arguments for instantiating Vanilla LSTM Model.')
            
        seq_length = model_params_dict['input_width']
        n_features = model_params_dict['n_features']
        neurons_l1 = model_params_dict['neurons_l1']
        neurons_out = model_params_dict['neurons_out']
        dropout_l1 = model_params_dict['dropout_l1']
        activationFn_1 = model_params_dict['act_l1']
        isDropout = model_params_dict['isDropout']
        
        return Vanilla_Model(seq_length,n_features,neurons_l1,neurons_out,
                             dropout_l1,activationFn_1,isDropout)
    
    def __Get_Stacked_LSTM_Model(self,model_params_dict):
        
        args_list = list(['input_width','n_features','neurons_l1','neurons_l2','neurons_out',
                          'act_l1','act_l2','act_l3'])
        
        if all(key in args_list for key in model_params_dict.keys()) == False:
            raise ValueError('Missing arguments for instantiating Stacked LSTM Model.')
            
        seq_length = model_params_dict['input_width']
        n_features = model_params_dict['n_features']
        neurons_l1 = model_params_dict['neurons_l1']
        neurons_l2 = model_params_dict['neurons_l2']        
        neurons_out = model_params_dict['neurons_out']       
        activationFn_1 = model_params_dict['act_l1']
        activationFn_2 = model_params_dict['act_l2']
        activationFn_3 = model_params_dict['act_l3']       
        
        return Stacked_Model(seq_length,n_features,neurons_l1,neurons_l2,neurons_out,
                             activationFn_1,activationFn_2,activationFn_3)
    
        
    def __Get_Custom_Model(self,model_params_dict):
        
        args_list = list(['input_shape','neurons_l1','neurons_l2','neurons_out',
                              'act_l1','act_l2'])
        
        if all(key in args_list for key in model_params_dict.keys()) == False:
            raise ValueError('Missing arguments for instantiating Custom Model.')
        
        input_shape = model_params_dict['input_shape']
        neurons_l1 = model_params_dict['neurons_l1']
        neurons_l2 = model_params_dict['neurons_l2']
        neurons_out = model_params_dict['neurons_out'] 
        act_l1 = model_params_dict['act_l1']
        act_l2 = model_params_dict['act_l2']
        
        return Custom_Model(input_shape,
                                neurons_l1,
                                neurons_l2,
                                neurons_out,
                                act_l1,
                                act_l2)
        
        
        
        