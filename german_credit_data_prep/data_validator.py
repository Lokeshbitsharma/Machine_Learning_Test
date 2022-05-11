# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import configparser

from data_validation.data_properties import DataPropertyConfig
from data_preparation.feature_meta_data import FeatureMetaData

from pandas_schema import Column, Schema
from pandas_schema.validation import CustomElementValidation,InListValidation,IsDtypeValidation,InRangeValidation

from constants import Constants

class DataValidator:
    
    def __init__(self):
        self.feature_meta_data = FeatureMetaData()
        self.schema = self.__schema_generator()
        
        
    def __schema_generator(self):
        '''
        Returns
        -------
        val_obj : Schema for data validation
            It helps to validate data when called for validation

        '''
        data_property_config = DataPropertyConfig()
        self.data_property = data_property_config.fetch_dataproperty_from_config()
        
        schema_dict = dict()
        
        for feature in self.feature_meta_data.INT_COLUMNS:            
            schema_dict[feature] = [IsDtypeValidation(self.data_property[feature]['type']),
                                    InRangeValidation(self.data_property[feature]['min'], self.data_property[feature]['max'] + 1)]
            
            
        for feature in self.feature_meta_data.BOOL_COLUMNS:            
            schema_dict[feature] = [IsDtypeValidation(self.data_property[feature]['type']),
                                    InListValidation(self.data_property[feature]['value'])]
            
        for feature in self.feature_meta_data.STR_COLUMNS:            
            schema_dict[feature] = [InListValidation(self.data_property[feature]['value'])]
        
        schema_element = list()
        for feature in self.feature_meta_data.FEATURE_COLUMNS:
            schema_element.append(Column(feature, schema_dict[feature]))
                
        return Schema(schema_element)
    
    def validate(self,data):
        '''
        
        Parameters
        ----------
        data : pandas dataframe
            DESCRIPTION.

        Returns
        -------
        If validation successful, return true. Else return errors.
            DESCRIPTION.

        '''     
        errors = self.schema.validate(data)
        
        if len(errors) == 0:
            return True
        else:
            return errors
        