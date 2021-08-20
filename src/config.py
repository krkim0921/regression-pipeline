TRAINING_CSV = "inputs/train.csv"
TRAINING_FOLDS_CSV = "inputs/train_folds.csv"

TARGET = 'SalePrice'

# input var
FEATURES = ['MSSubClass', 'MSZoning', 'Neighborhood',
            'OverallQual', 'OverallCond', 'YearRemodAdd',
            'RoofStyle', 'MasVnrType', 'BsmtQual', 'BsmtExposure',
            'HeatingQC', 'CentralAir', '1stFlrSF', 'GrLivArea',
            'BsmtFullBath', 'KitchenQual', 'Fireplaces', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageCars', 'PavedDrive',
            'LotFrontage',
            # temporal variable:
            'YrSold']


# Drop feature after calculation
DROP_FEATURES = 'YrSold'

# numerical var with NA 
NUMERICAL_VARS_WITH_NA = ['LotFrontage']

# categorical var with NA 
CATEGORICAL_VARS_WITH_NA = ['MasVnrType', 'BsmtQual', 'BsmtExposure',
                            'FireplaceQu', 'GarageType', 'GarageFinish']

TEMPORAL_VARS = 'YearRemodAdd'

# Vars need distribution transform
NUMERICALS_LOG_VARS = ['LotFrontage', '1stFlrSF', 'GrLivArea']

# categorical var to encode
CATEGORICAL_VARS = ['MSZoning', 'Neighborhood', 'RoofStyle', 'MasVnrType',
                    'BsmtQual', 'BsmtExposure', 'HeatingQC', 'CentralAir',
                    'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish',
                    'PavedDrive']


# Reproducible Random State
RANDOME_STATE = 777

# Weights Directories
TRAIN_WEIGHTS_DIR = "weights/"

#Prediction Directories
PRED_PIPE_DIR = "weights/12:27:42"
PIPELINE_NAME = "et_trained_pipeline"
