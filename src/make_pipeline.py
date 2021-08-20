
from .preprocessing.cat_preprocessor import CategoricalEncoder, CategoricalMissing, RareLabelEncoder
from .preprocessing.numeric_preprocessor import NumericalMissing, LogTransformer
from .preprocessing.temp_preprocessor import DateFeatureCreater, DropFeature

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from . import config
from .model_dispatcher import REG_MODELS

def train_pipeline(model):
    house_price_pipeline = Pipeline(
        [
            (
                "numric_missing_imputation",
                NumericalMissing(
                    variables= config.NUMERICAL_VARS_WITH_NA, mode = 'median')),
            (
                "cat_missing_imputation",
                CategoricalMissing(
                    variables= config.CATEGORICAL_VARS_WITH_NA)),
            
            (
                "date_convert",
                DateFeatureCreater(variables=config.TEMPORAL_VARS, ref_var=config.DROP_FEATURES)),

            (
               "unique_label_convert",
               RareLabelEncoder(variables=config.CATEGORICAL_VARS, proportion=0.05)),
            
            (
                "categorical_encode",
                CategoricalEncoder(variables=config.CATEGORICAL_VARS)),
            (
                "log_transformation",
                LogTransformer(variables=config.NUMERICALS_LOG_VARS)),

            (
                "drop_feature",
                DropFeature(variables=config.DROP_FEATURES)),

            (
                'scaler', MinMaxScaler()),

            (
                f'{model} selected', REG_MODELS[model]
            ),           
        ]
    )
    return house_price_pipeline

