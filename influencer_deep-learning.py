import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import LSTM, Flatten
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler

x_train = pd.read_csv('influencer_train_log.csv', header = 0)
y_train = pd.read_csv('influencer_label.csv', header = 0)
x_test = pd.read_csv('influencer_test_log.csv', header = 0)

x_train = np.asarray(x_train)
y_train= np.asarray(y_train)
x_test = np.asarray(x_test)

# prep data for CNN
x_train_cnn = x_train.reshape(5500,22,1)
x_test_cnn = x_test.reshape(5952,22,1)

input_features = 22

#CNN
cnn = Sequential()
cnn.add(Conv1D(512, 6, activation="relu", input_shape=(input_features,1)))
cnn.add(Conv1D(512, 6, activation="relu"))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Dropout(0.25))
cnn.add(Conv1D(128, 3, activation="relu"))
cnn.add(Conv1D(256, 3, activation="relu"))
# cnn.add(MaxPooling1D(pool_size=2))
cnn.add(GlobalAveragePooling1D())
cnn.add(Dropout(0.25))
#cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(1, activation='sigmoid'))

cnn.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

cnn.fit(x_train_cnn, y_train, batch_size=512, epochs=20)
y_pred_cnn = cnn.predict_proba(x_test_cnn, batch_size=256)

y_pred_cnn.tofile('cnn sub 4.csv',sep=',',format='%10.5f')

#MLP
mlp = Sequential()
mlp.add(Dense(256, activation='relu', input_dim= input_features))
mlp.add(Dense(128, activation='relu'))
mlp.add(Dropout(0.25))
mlp.add(Dense(32, activation='relu'))
mlp.add(Dropout(0.5))
mlp.add(Dense(256, activation='relu'))
mlp.add(Dense(1, activation='sigmoid'))

mlp.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
              
mlp.fit(x_train, y_train, batch_size=512, epochs=20)
y_pred_mlp = mlp.predict_proba(x_test, batch_size=256)
y_pred_mlp.tofile('mlp sub 1.csv',sep=',',format='%10.5f')
