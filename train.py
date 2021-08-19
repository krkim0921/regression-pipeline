import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from src.make_pipeline import train_pipeline
import src.config as config

def run_train(model):
    '''
    Let's Rock'n Roll : train the model
    '''
    print('Training Is Starting....PRAY ')

    df = pd.read_csv(config.TRAINING_CSV)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[config.FEATURES],
        df[config.TARGET],
        test_size = 0.1,
        random_state = config.RANDOME_STATE
        
    )

    # transform Target
    y_trian = np.log(y_train)
    print(y_train.shape)
    print(X_train.shape)

    house_pipeline = train_pipeline(model)
    house_pipeline.fit(X_train, y_train)

if __name__ == '__main__':
    run_train('rf')

