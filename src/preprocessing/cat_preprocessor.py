'''
Author: Ian Kim
Date: 2021.08.19
Licence: AitheNutrigene

'''
             #         #######               
            ###           #
           #####          #  
         ########      ########     #

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalMissing(BaseEstimator, TransformerMixin):

    '''
    This class is for categorical missing imputer
    This class find missing values in categorical columns 
    and impute as "Missing"

    Args:
        [
        variables: List of columns(categorical columns), type: List
        ]
    '''
    def __init__(self, variables=None):
        
        # check list or not
        if not isinstance(variables, list):
            self.variables = list(variables)
        else:
            self.variables = variables

        # Mandatory for sklearn pipeline
    def fit(self, X, y = None):
        return self

        #loop through and fill N/A value as "Missing"
    def transform(self, X, y = None):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].fillna('Missing')

        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables = None):
        
        if not isinstance(variables, list):
            self.variables = list(variables)
        else:
            self.variables = variables

    def fit(self, X, y):
        
        temp_df = pd.concat([X, y], axis = 1)
        temp_df.columns = list(X.columns) + ['target']

        self.encode_dict = dict()

        for col in self.variables:
            self.encode_dict[col] = temp_df.groupby([col])['target'].mean().to_dict()

        return self

    def transform(self, X, y = None):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].map(self.encode_dict[col])

        return X



class RareLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, variables, proportion = 0.05):

        if not isinstance(variables, list):
            raise ValueError('Input variables should be List type')

        self.variables = variables
        self.proportion =proportion

    
    # find Rare label and convert to most frequent Label
    def fit(self, X, y = None):
        self.encoder_dict = dict()

        for col in self.variables:
            t = pd.Series(X[col].value_counts(normalize = True))
            self.encoder_dict[col] = list(t[t >= self.proportion].index)

        return self

    def transform(self, X, y = None):

        X = X.copy()
        for col in self.variables:
            X[col] = np.where(X[col].isin(self.encoder_dict[
                    col]), X[col], 'Rare')
            

        return X
            
            
        
