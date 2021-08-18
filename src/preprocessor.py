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

    def __init__(self, variables=None) -> None:
        
        # check list or not
        if not isinstance(variables, list):
            self.variables = list(variables)
        else:
            self.variables = variables

        # Mandatory for sklearn pipeline
        def fit(self, X, y = None) -> None:
            return self

        def transform(self, X):
            X = X.copy()
            for col in self.variables:
                X[col] = X[col].fillna('Missing')

            return X

class NumericalMissing(BaseEstimator, TransformerMixin):

    def __init__(self, variables= None, mode = 'frequent') -> None:
        
        self.mode = mode
        
        if not isinstance(variables, list):
            self.variables = list(variables)
        else:
            self.variables = variables

    def fit(self, X, y=None) ->None:
        self.imputer = dict()

        if self.mode == 'frequent':
            for col in self.variables:
                self.imputer[col] = X[col].mode()[0]

        elif self.mode == 'average':
            for col in self.variables:
                self.imputer[col] = X[col].mean()

        elif self.mode == 'median':
            for col in self.variables:
                self.imputer[col] = X[col].median()   
        else:
            raise ValueError('Valid option: frequent, average, median')
                
        return self

    def transform(self, X, y = None)-> None:

        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.imputer[col], inplace = True)
        return X
        

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables = None) -> None:
        
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

    def transform(self, X, y):
        X = X.copy()
        for col in self.variables:
            X[col] = X[col].map(self.encode_dict[col])

        return X

