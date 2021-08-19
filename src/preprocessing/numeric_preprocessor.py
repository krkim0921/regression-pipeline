
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NumericalMissing(BaseEstimator, TransformerMixin):

    def __init__(self, variables= None, mode = 'frequent') -> None:
        
        self.mode = mode
        
        if not isinstance(variables, list):
            self.variables = list(variables)
        else:
            self.variables = variables

    def fit(self, X, y=None):
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

    def transform(self, X, y = None):

        X = X.copy()
        for col in self.variables:
            X[col].fillna(self.imputer[col], inplace = True)
        return X


# Logarithm Distribution Converter 
class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X, y = None):
        X = X.copy()

        for col in self.variables:
            X[col] = np.log(X[col])

        return X

