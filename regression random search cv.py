import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from numpy.random import random_sample

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
# train data
train_df = pd.read_csv('train.csv', header=0)
cols_to_transform = train_df[['X0', 'X1', 'X2','X3', 'X4', 'X5', 'X6', 'X8']]
train_df= pd.get_dummies(data=train_df, columns = cols_to_transform )
x_train = train_df.drop(train_df.columns[[0, 1]], axis=1).values
y_train = train_df[['y']].values

x_train = np.asarray(x_train)
y_trian = np.asarray(y_train)

test_df = pd.read_csv('test.csv', header=0)
y_test = test_df[['y']]

rf = RandomForestRegressor(n_estimators= 500)

random_float = random_sample()
param_dist = {#"n_estimators": sp_randint(300,750),
              "max_depth": [3, None],
              "max_features": sp_randint(1, 20),
              #"min_samples_split": [2, random_float],
              "min_samples_leaf": sp_randint(1, 20),
              "bootstrap": [True, False],
              "criterion": ["mse", "mae"]}

n_iter_search = 20
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
                                   
random_search_rf.fit(x_train, np.ravel(y_train))
report(random_search_rf.cv_results_)