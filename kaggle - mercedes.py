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
from keras.callbacks import EarlyStopping
from keras import regularizers
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe



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

              

