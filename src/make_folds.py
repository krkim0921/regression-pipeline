import pandas as pd
from sklearn import model_selection
import os


if __name__ == '__main__':
    path = ('inputs/')
    df = pd.read_csv(os.path.join(path, 'train.csv')) # (1460, 81)
    
    # add new col for 'kfold'
    df['kfold'] = -1

    #sampling it
    df = df.sample(frac=1).reset_index(drop= True) # (1460, 82)
    y = df.SalePrice.values

    skfold = model_selection.StratifiedKFold(n_splits=5)
    for f, (t, v) in enumerate(skfold.split(X = df, y = y)):
        df.loc[v, 'kfold'] = f

    df.to_csv(os.path.join(path, 'train_folds.csv'), index = False)

