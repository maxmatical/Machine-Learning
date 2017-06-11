import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import LSTM, Flatten
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, BayesianRidge, Ridge
from keras.callbacks import EarlyStopping
from keras import regularizers
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from numpy.random import random_sample


# train data
train_df = pd.read_csv('train.csv', header=0)
cols_to_transform = train_df[['X0', 'X1', 'X2','X3', 'X4', 'X5', 'X6', 'X8']]
train_df= pd.get_dummies(data=train_df, columns = cols_to_transform )
x_train = train_df.drop(train_df.columns[[0, 1]], axis=1).values
y_train = train_df[['y']].values

x_train = np.asarray(x_train)
y_trian = np.asarray(y_train)

#quick checks
type(x_train)
type(y_train)
x_train.shape
y_train.shape

#test data
test_df = pd.read_csv('test.csv', header=0)
y_test = test_df[['y']]

#PCA
pca = PCA(n_components=0.8, whiten=True)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_train_pca.shape

#fitting a MLP 
nrows_full = 563
nrows_pca = 46
model = Sequential()
model.add(Dense(256, input_dim = nrows_pca, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
# model.add(Dropout({{uniform(0, 1)}}))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(optimizer='adam',
              loss='mean_squared_error')
              
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
model.fit(x_train_pca, y_train, validation_split=0.2, callbacks=[early_stopping],
            epochs=20, batch_size = 512)

model.predict(x_train_pca)        

# ensemble methods
erf = ExtraTreesRegressor()
rf = RandomForestRegressor(n_estimators= 500)
gb = GradientBoostingRegressor(n_estimators=500)
#random search for optimizing hyperparams

random_float = random_sample()
param_dist = {"n_estimators": sp_randint(300,750),
              #"max_depth": [3, None],
              #"max_features": sp_randint(1, 20),
              #"min_samples_split": [2, random_float],
              #"min_samples_leaf": sp_randint(1, 20),
              #"bootstrap": [True, False],
              "criterion": ["mse", "mae"]}
param_dist_gb = {"n_estimators": sp_randint(300,750),
              "max_depth": [3, None],
              "max_features": sp_randint(1, 20),
              "min_samples_split": sp_randint(1, 20),
              "min_samples_leaf": sp_randint(1, 20),
              #"bootstrap": [True, False],
              "criterion": ["friedman_mse", "mae"]}
              
n_iter_search = 20
random_search_erf = RandomizedSearchCV(erf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
                                   
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
random_search_gb = RandomizedSearchCV(gb, param_distributions=param_dist_gb,
                                   n_iter=n_iter_search)

random_search_erf.fit(x_train, np.ravel(y_train))
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search_erf.cv_results_)
