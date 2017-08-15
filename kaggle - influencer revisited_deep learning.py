import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.optimizers import RMSprop
from imblearn.over_sampling import SMOTE, ADASYN 
from sklearn.preprocessing import StandardScaler, Normalizer
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV



df = pd.read_csv('influencer_train.csv', header = 0)
df = np.asarray(df).astype(np.float32)
y = np.ravel(df[:,0])
x = df[:,1:23]
x_test = pd.read_csv('influencer_test.csv', header = 0)
np.asarray(x_test)

# upsampling
##########################
x_resampled, y_resampled = SMOTE().fit_sample(x,y)
xtr, xtt, ytr, ytt = train_test_split(x_resampled, y_resampled, test_size = 0.2)
xtr.astype(np.float32)
xtt.astype(np.float32)
ytr.astype(np.float32)
ytt.astype(np.float32)

# defining input shape
##########################
n_rows = xtr.shape[1]
# early stopping
##########################
early_stopping = EarlyStopping(monitor='val_loss', patience=4)  

#models
##########################
# basic 1 hidden layer, wide
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
          
history1 = model1.fit(xtr, ytr, validation_split=0.2, callbacks=[early_stopping],
            epochs=50, batch_size= 32)

# model 2: few hidden units, deep
model2 = Sequential()
model2.add(Dense(64, input_dim = n_rows,
                activation = 'relu',
                kernel_initializer = 'uniform',
                kernel_regularizer = regularizers.l2(0.01)))
model2.add(Dropout(0.5))
model2.add(Dense(64, 
                activation = 'relu',
                kernel_initializer = 'uniform',
                kernel_regularizer = regularizers.l2(0.01)))
model2.add(Dropout(0.5))
model2.add(Dense(64, 
                activation = 'relu',
                kernel_initializer = 'uniform',
                kernel_regularizer = regularizers.l2(0.01)))
model2.add(Dropout(0.5))
model2.add(Dense(64, 
                activation = 'relu',
                kernel_initializer = 'uniform',
                kernel_regularizer = regularizers.l2(0.01)))
model2.add(Dropout(0.5))
model2.add(Dense(64,
                activation = 'relu',
                kernel_initializer = 'uniform',
                kernel_regularizer = regularizers.l2(0.01)))
model2.add(Dropout(0.5))
model2.add(Dense(1, activation='sigmoid'))
#compile
model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) 
# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=4)            
model2.fit(xtr, ytr, validation_split=0.2, callbacks=[early_stopping],
            epochs=20, batch_size= 32)

#model3, less hidden layers than model2
model3 = Sequential()
model3.add(Dense(64, input_dim = n_rows,
                activation = 'relu',
                kernel_initializer = 'uniform',
                kernel_regularizer = regularizers.l2(0.01)))
model3.add(Dropout(0.5))
model3.add(Dense(64, 
                activation = 'relu',
                kernel_initializer = 'uniform',
                kernel_regularizer = regularizers.l2(0.01)))
model3.add(Dropout(0.5))
# model3.add(Dense(64, 
#                 activation = 'relu',
#                 kernel_initializer = 'uniform',
#                 kernel_regularizer = regularizers.l2(0.01)))
# model3.add(Dropout(0.5))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) 
model3.fit(xtr, ytr, validation_split=0.2, callbacks=[early_stopping],
            epochs=20, batch_size= 32)
            
            
# model4, experiment with different initializers
model4 = Sequential()
model4.add(Dense(64, input_dim = n_rows,
                activation = 'relu',
                kernel_initializer = 'he_normal',
                kernel_regularizer = regularizers.l2(0.01)))
model4.add(Dropout(0.5))
model4.add(Dense(64, 
                activation = 'relu',
                kernel_initializer = 'he_normal',
                kernel_regularizer = regularizers.l2(0.01)))
model4.add(Dropout(0.5))
model4.add(Dense(1, activation='sigmoid'))
model4.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) 
history4 = model4.fit(xtr, ytr, validation_split=0.2, callbacks=[early_stopping],
            epochs=50, batch_size= 32)

# model5: different structure (shrinking hidden units)
model5 = Sequential()
model5.add(Dense(128, input_dim = n_rows,
                activation = 'relu',
                kernel_initializer = 'he_normal',
                kernel_regularizer = regularizers.l2(0.01)))
model5.add(Dropout(0.5))
model5.add(Dense(64, 
                activation = 'relu',
                kernel_initializer = 'he_normal',
                kernel_regularizer = regularizers.l2(0.01)))
model5.add(Dropout(0.5))
model5.add(Dense(1, activation='sigmoid'))
model5.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']) 
history5 = model5.fit(xtr, ytr, validation_split=0.2, callbacks=[early_stopping],
            epochs=50, batch_size= 32)




# accuracy of models
##########################
accuracy_score(ytt, model1.predict(xtt).astype(int))
# model 1, around 75% with adam, seems better than rmsprop
accuracy_score(ytt, model2.predict(xtt).astype(int))
# model 2, 47.7% with 5 hidden layers (8 hidden units), 50.8% with 64 units 
# from model2, it looks like relu/selu doesn't matter
accuracy_score(ytt, model3.predict(xtt).astype(int)) 
# 2 layers seems to work better than 3 layers, best so far
# tried using selu instead of relu, and adding batchnormalization layers, accuracy did not improve
accuracy_score(ytt, model4.predict(xtt).astype(int)) 
# using glorot normal => 0.75646743978590547 accuracy, very similar to uniform
# using glorot uniform => poor performance
# he_normal => 76.3%, a bit better
# experimented with number of hidden units, looks like 64, 64 is best
accuracy_score(ytt, model5.predict(xtt).astype(int)) 
# model 5 worked also pretty well


# checking history of model
##########################
history = history5

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# grid search for tuning hyperparameters of the models
def create_model(optimizer = 'adam'):
    model4 = Sequential()
    model4.add(Dense(64, input_dim = n_rows,
                    activation = 'relu',
                    kernel_initializer = 'he_normal',
                    kernel_regularizer = regularizers.l2(0.01)))
    model4.add(Dropout(0.5))
    model4.add(Dense(64, 
                    activation = 'relu',
                    kernel_initializer = 'he_normal',
                    kernel_regularizer = regularizers.l2(0.01)))
    model4.add(Dropout(0.5))
    model4.add(Dense(1, activation='sigmoid'))
    model4.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy']) 
    return model4

model4 = KerasClassifier(build_fn=create_model, epochs=60, batch_size=32, verbose=0)
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model4, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(xtr, ytr)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
