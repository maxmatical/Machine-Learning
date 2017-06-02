import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import LSTM, Flatten, Reshape
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler
from sklearn.metrics import accuracy_score

x = pd.read_csv('ret_full.csv', header=0)
y = pd.read_csv('roro_full.csv', header=0)
x = np.asarray(x)
y = np.asarray(y)
n1 = np.int(len(y)*0.6)
n2 = np.int(len(y)*0.7)
n3 = np.int(len(y)*0.8)

x_train_1 = x[:n1]
x_train_2 = x[:n2]
x_train_3 = x[:n3]
y_train_1 = y[:n1]
y_train_2 = y[:n2]
y_train_3 = y[:n3]
x_test_1 = x[n1:]
y_test_1 = y[n1:]

x_test = pd.read_csv('ret_test.csv', header=0)
y_test = pd.read_csv('roro_test.csv', header=0)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

x_train_cnn_1 = x_train_1.reshape(n1,26,1)
x_train_cnn_2 = x_train_2.reshape(n2,26,1)
x_train_cnn_3 = x_train_3.reshape(n3,26,1)
x_test_cnn = x_test.reshape(260,26,1)
x_test_cnn_1 = x_test_1.reshape(520,26,1)

x_train_lstm_1 = x_train_1.reshape(n1,1,26)
x_train_lstm_2 = x_train_2.reshape(n2,1,26)
x_train_lstm_3 = x_train_3.reshape(n3,1,26)
x_test_lstm = x_test.reshape(260, 1, 26)


#CNN Model
cnn = Sequential()
cnn.add(Conv1D(512, 6, activation="relu", input_shape=(26,1)))
cnn.add(Conv1D(512, 6, activation="relu"))
cnn.add(Dropout(0.25))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Conv1D(128, 3, activation="relu"))
cnn.add(Conv1D(256, 3, activation="relu"))
cnn.add(Dropout(0.25))
# cnn.add(MaxPooling1D(pool_size=2))
cnn.add(GlobalAveragePooling1D())
cnn.add(Dropout(0.5))
#cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dense(1, activation='sigmoid'))

cnn.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

cnn.fit(x_train_cnn_1, y_train_1, batch_size=512, epochs=5)
cnn_pred_1 = cnn.predict_proba(x_test_cnn, batch_size=128)
cnn.fit(x_train_cnn_2, y_train_2, batch_size=512, epochs=5)
cnn_pred_2 = cnn.predict_proba(x_test_cnn, batch_size=128)
cnn.fit(x_train_cnn_3, y_train_3, batch_size=512, epochs=5)
cnn_pred_3 = cnn.predict_proba(x_test_cnn, batch_size=128)
cnn_avg_pred = (cnn_pred_1+cnn_pred_2+cnn_pred_3)/3
cnn_avg_pred = np.round(cnn_avg_pred)
accuracy_score(cnn_avg_pred, y_test)

cnn.evaluate(x_test_cnn, y_test)

#lstm
model = Sequential()
model.add(Embedding(26, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train_1, y_train_1, batch_size=350, epochs=5)
model.evaluate(x_test, y_test, batch_size=150)

# Stacked LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(1, 26))) 
model.add(LSTM(64, return_sequences=True))  
model.add(Dropout(0.25))
model.add(LSTM(32)) 
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
              

model.fit(x_train_lstm_1, y_train_1, batch_size=512, epochs=10)
lstm_pred_1 = model.predict_proba(x_test_lstm, batch_size=128)
model.fit(x_train_lstm_2, y_train_2, batch_size=512, epochs=10)
lstm_pred_2 = model.predict_proba(x_test_lstm, batch_size=128)
model.fit(x_train_lstm_3, y_train_3, batch_size=512, epochs=10)
lstm_pred_3 = model.predict_proba(x_test_lstm, batch_size=128)
lstm_avg_pred = (lstm_pred_1+lstm_pred_2+lstm_pred_3)/3
lstm_avg_pred = np.round(lstm_avg_pred)
accuracy_score(lstm_avg_pred, y_test)

model.evaluate(x_test_lstm, y_test, batch_size=128)

#LSTM + CNN
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(1, 26))) 
model.add(LSTM(64, return_sequences=True))  
model.add(Dropout(0.25))
model.add(LSTM(32)) 
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Reshape((26,1)))
model.add(Conv1D(512, 2, activation="relu", input_shape=(26,1)))
cnn.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

