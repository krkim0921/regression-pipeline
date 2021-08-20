from sklearn import ensemble

REG_MODELS = {
    'rf': ensemble.RandomForestRegressor(max_depth=5, n_jobs=1, verbose=2),
    'et': ensemble.ExtraTreesRegressor(n_estimators=200, n_jobs=1, verbose=2),
    'ada': ensemble.AdaBoostRegressor(n_estimators=100, learning_rate=0.1),
}