import pandas as pd
import numpy as np
#import quandl
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score

x_train = pd.read_csv('ret_train.csv', header = 0)
y_train = pd.read_csv('roro_train.csv', header = 0)
x_test = pd.read_csv('ret_test.csv', header = 0)
y_test = pd.read_csv('roro_test.csv', header = 0)

#classifiers
# KNN
knn_clf = KNeighborsClassifier(n_neighbors=10)
knn_clf.fit(x_train, np.ravel(y_train))
y_pred = knn_clf.predict(x_test)
accuracy_score(y_test, y_pred)

#logistic regression
lr_clf = LogisticRegression(penalty='l1', C=1, max_iter=10000)
lr_clf.fit(x_train, np.ravel(y_train))
y_pred = lr_clf.predict(x_test)
accuracy_score(y_test, y_pred)

# random forest
forest_clf = RandomForestClassifier(n_estimators=500, max_depth=15)
forest_clf.fit(x_train, np.ravel(y_train))
y_pred_fr = forest_clf.predict(x_test)
accuracy_score(y_test, y_pred_fr)

#sklearn gb
gb_clf = GradientBoostingClassifier()
gb_clf.fit(x_train, np.ravel(y_train))
y_pred_gb = gb_clf.predict(x_test)
accuracy_score(y_test, y_pred_gb)

# xgb
xgb_clf = xgb.XGBClassifier(subsample=0.5, n_estimators= 300, max_depth=0)
xgb_clf.fit(x_train, np.ravel(y_train))
y_pred_xgb = xgb_clf.predict(x_test)
accuracy_score(y_test, y_pred_xgb)

