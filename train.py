'''
Author: Ian Kim
Date: 2021.08.19
Licence: AitheNutrigene

'''
             #         #######               
            ###           #
           ## ##          #  
         ###   ###      ########     #


import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn import metrics
from src.make_pipeline import train_pipeline
import src.config as config
import utils



def run_train(model):
    #Let's Rock'n Roll : train the model
    print('Training Is Starting....PRAY ')

    df = pd.read_csv(config.TRAINING_CSV)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[config.FEATURES],
        df[config.TARGET],
        test_size = 0.1,
        random_state = config.RANDOME_STATE )

    # transform Target
    y_train = np.log(y_train)

    #train
    house_pipeline = train_pipeline(model)
    house_pipeline.fit(X_train[config.FEATURES], y_train)

    #predict
    preds = house_pipeline.predict(X_test[config.FEATURES])

    print('-----------------Result-----------------')
    print(np.sqrt(metrics.mean_squared_error(y_test, preds)))
    #save pipeline
    print('---------Model Saved Completed------------')
    utils.save_pipeline(house_pipeline, model)
    
    print(      'Training Is Finished!!')
    print('🤩😎🥳😍🤗🤩😎🥳😍🤗🤩😎🥳😍🤗🤩😎🥳😍🤗🤩😎🥳😍🤗')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose the Model for Regression Problem')
    parser.add_argument('--model', type = str, required=True, help='check avaliable models in dispatcher.py')
    args = parser.parse_args()
    
    run_train(args.model)



    

