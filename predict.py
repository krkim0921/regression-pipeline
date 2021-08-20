import pandas as pd
import utils
from src import config
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np


def make_prediction(input_data):
    
    model = utils.load_model(path=config.PRED_PIPE_DIR, pipeline_name=config.PIPELINE_NAME)
    preds = model.predict(input_data)

    return preds


if __name__ == '__main__':
    df = pd.read_csv(config.TRAINING_CSV)
    X_train, X_test, y_train, y_test = train_test_split(
    df[config.FEATURES],
    df[config.TARGET],
    test_size=0.1,
    random_state=config.RANDOME_STATE)

    preds = make_prediction(X_test[config.FEATURES])

    print(f'Test Result: {np.sqrt(metrics.mean_squared_error(y_test, preds))}')
    

