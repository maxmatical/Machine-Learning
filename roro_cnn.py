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



x_train = pd.read_csv('ret_train.csv', header=0)
y_train = pd.read_csv('roro_train.csv', header=0)
# x_test = pd.read_csv('ret_test.csv', header=0)
# y_test = pd.read_csv('roro_test.csv', header=0)
x_live = pd.read_csv('roro_factorsP-NEW1.csv', header=0)
y_live = pd.read_csv('roro_bin-NEW.csv', header=0)

x_train = np.asarray(x_train)
y_train= np.asarray(y_train)
# x_test = np.asarray(x_test)
# y_test = np.asarray(y_test
x_live = np.asarray(x_live)
y_live = np.asanyarray(y_live)



# scaler = MinMaxScaler()
# scaler = Normalizer()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test=  scaler.transform(x_test)

x_train_cnn = x_train.reshape(1040,26,1)
# x_test_cnn = x_test.reshape(260,26,1)
x_live_cnn = x_live.reshape(24,26,1)

x_train_lstm = x_train.reshape(1040,1,26)
# x_test_lstm = x_test.reshape(260, 1, 26)
x_live_lstm = x_live.reshape(24,1,26)

#MLP
#create model
model = Sequential()
model.add(Dense(256, input_dim=26, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
#compile
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
#train
model.fit(x_train, y_train, epochs=20, batch_size=512)
live_pred = model.predict(x_live)
accuracy_score(live_pred, y_live)

#CNN Model
cnn = Sequential()
cnn.add(Conv1D(512, 6, activation="relu", input_shape=(26,1)))
cnn.add(Conv1D(512, 6, activation="relu"))
cnn.add(Dropout(0.25))
cnn.add(MaxPooling1D(pool_size=2))
cnn.add(Conv1D(128, 3, activation="relu"))
cnn.add(Conv1D(3256, 3, activation="relu"))
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

cnn.fit(x_train_cnn, y_train, batch_size=512, epochs=10)
# cnn.evaluate(x_test_cnn, y_test)
live_pred = cnn.predict(x_live_cnn)
accuracy_score(live_pred, y_live)


#LSTM
model = Sequential()
model.add(Embedding(26, output_dim=256))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=512, epochs=20)
# model.evaluate(x_test, y_test, batch_size=128)
live_pred = model.predict(x_live)
accuracy_score(live_pred, y_live)

# Stacked LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(1, 26))) 
model.add(Dropout(0.25))
model.add(LSTM(64, return_sequences=True))  
model.add(LSTM(32)) 
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
              
model.fit(x_train_lstm, y_train, batch_size=512, epochs=20)
live_pred = model.predict(x_live_lstm)
accuracy_score(live_pred, y_live)

#model.evaluate(x_test_lstm, y_test, batch_size=150)


