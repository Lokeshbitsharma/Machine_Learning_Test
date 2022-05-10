# -*- coding: utf-8 -*-
"""
Feature Meta Data
"""
import configparser
import ast
from constants import Constants

class FeatureMetaData:
    
    def __init__(self):
        config_path = Constants.config_path
        self.parser = configparser.ConfigParser()
        self.parser.read(config_path)
        self.generate_feature_from_config()
        
    def generate_feature_from_config(self):
        '''
        Generate Input Features, Labels, AllColumns, Data type specific feature list form config file.
        Returns
        -------
        None.

        '''
        self.LABEL_COLUMN = ast.literal_eval(self.parser.get("feature_specific","LABEL_COLUMN"))    
        self.BOOL_COLUMNS = ast.literal_eval(self.parser.get("feature_specific","BOOL_COLUMNS"))
        self.INT_COLUMNS = ast.literal_eval(self.parser.get("feature_specific","INT_COLUMNS"))
        self.STR_COLUMNS = ast.literal_eval(self.parser.get("feature_specific","STR_COLUMNS"))
        self.STR_NUNIQUESS = ast.literal_eval(self.parser.get("feature_specific","STR_NUNIQUESS"))
        self.FLOAT_COLUMNS = ast.literal_eval(self.parser.get("feature_specific","FLOAT_COLUMNS"))

        self.FEATURE_COLUMNS = (self.INT_COLUMNS + self.BOOL_COLUMNS + self.STR_COLUMNS + self.FLOAT_COLUMNS)
        self.ALL_COLUMNS = self.FEATURE_COLUMNS + [self.LABEL_COLUMN]
        