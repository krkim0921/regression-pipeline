from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class DateFeatureCreater(BaseEstimator, TransformerMixin):
    
    def __init__(self, variables, ref_var):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables
  
        self.ref_var = ref_var

    def fit(self, X, y=None):
        return self


    def transform(self, X, y = None):
        X = X.copy()

        for col in self.variables:
            X[col] = X[self.ref_var] - X[col]

        return X

class DropFeature(BaseEstimator, TransformerMixin):

    def __init__(self, variables):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y= None):
        return self

    def transform(self, X, y = None):
        X = X.copy()

        for col in self.variables:
            X = X.drop(col, axis = 1)

        return X
