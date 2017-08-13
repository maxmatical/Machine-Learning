import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.optimizers import RMSprop
from imblearn.over_sampling import SMOTE, ADASYN 
from sklearn.preprocessing import StandardScaler, Normalizer
df = pd.read_csv('influencer_train.csv', header = 0)
df = np.asarray(df).astype(np.float32)
y = np.ravel(df[:,0])
x = df[:,1:23]
x_test = pd.read_csv('influencer_test.csv', header = 0)
np.asarray(x_test)

# upsampling
x_resampled, y_resampled = SMOTE().fit_sample(x,y)
xtr, xtt, ytr, ytt = train_test_split(x_resampled, y_resampled, test_size = 0.2)


n_rows = xtr.shape[1]

# basic 1 hidden layer
model1 = Sequential()
# try different initializers, # of hidden unit,s activation function
model1.add(Dense(64, input_dim = n_rows, 
                activation='relu', 
                kernel_initializer = 'uniform', 
                kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.5)) # can experiment with dropout
model1.add(Dense(1, activation='sigmoid'))
# compile: try optimizers: adam, sgd, rmsprop
model1.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) 
# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=4)            
model1.fit(xtr, ytr, validation_split=0.2, callbacks=[early_stopping],
            epochs=20, batch_size= 32)

accuracy_score(ytt, model1.predict(xtt).astype(int)) #around 75% with adam, seems better than rmsprop

# model 2: add 1 more 


