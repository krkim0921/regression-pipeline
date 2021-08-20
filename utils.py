from sklearn.externals import joblib
import os
from datetime import datetime
from sklearn import pipeline

from sklearn.pipeline import Pipeline
from src.config import TRAIN_WEIGHTS_DIR

def save_pipeline(pipeline, model):
    time = datetime.now().strftime("%H:%M:%S")
    save_path = os.path.join(TRAIN_WEIGHTS_DIR, time)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_file_name = f'{model}_trained_pipeline'
    final_dir = os.path.join(save_path, save_file_name)
    joblib.dump(pipeline, final_dir)


def load_model(path, pipeline_name):

    if os.path.exists(path):
        model = joblib.load(filename = os.path.join(path, pipeline_name))
        print(model)
        return model
    else:
        print('check directory or pipeline_name')