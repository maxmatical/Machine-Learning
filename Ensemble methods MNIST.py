import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from lightgbm import LGBMClassifier

# loading data set - MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 1 example of the MNIST dataset
plt.imshow(X_train[0])

X_train.shape # 60000 28x28 pixel images
# want to reshape each flatten 28x28 to 1x784 
X_train = X_train.reshape(X_train.shape[0],  784)
X_test = X_test.reshape(X_test.shape[0],  784)

# preprocess and convert to grayscale by dividing by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#preprocess y vales
Y_train = np.ravel(y_train)
Y_test = np.ravel(y_test)
# start with simple decision tree
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
# lets see the accuracy of a simple decision tree
accuracy_score(Y_test, tree_clf.predict(X_test)) # 0.8749, not bad, but lets see what ensemble methods can do

# random forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, Y_train)
# accuracy_score
accuracy_score(Y_test, rf.predict(X_test)) # 0.9654, already much better

# adaboosting
ada = AdaBoostClassifier(n_estimators=100)
ada.fit(X_train, Y_train)
# accuracy_score
accuracy_score(Y_test, ada.predict(X_test)) # 0.6405, not as good

# gradient boosting - using lightgbm 
gbm = LGBMClassifier(n_estimators=100)
gbm.fit(X_train, Y_train)
# accuracy_score
accuracy_score(Y_test, gbm.predict(X_test)) # 0.9602

# Stacking - 0.5*rf+0.5*gbm
stack = VotingClassifier(estimators=[('rf', rf), ('gb', gbm)],
                        voting = 'soft')
stack.fit(X_train, Y_train)
# accuracy_score
accuracy_score(Y_test, stack.predict(X_test)) # 0.9651

# Stacking - 0.7*rf+0.3*gbm
stack2 = VotingClassifier(estimators=[('rf', rf), ('gb', gbm)],
                        voting = 'soft',
                        weights = [0.7, 0.3] )
stack2.fit(X_train, Y_train)
# accuracy_score
accuracy_score(Y_test, stack2.predict(X_test)) # 0.9665 wow a gain of 0.001 over the previous best classifier
# gain of 0.001 might seem low, but with 10000 predictions, that's 10 more accurately classified points

