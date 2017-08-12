import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, train_test_split

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

#train test split
xtr, xtt, ytr, ytt = train_test_split(x_train, y_train, test_size = 0.2)
ytr = np.ravel(ytr)
ytt = np.ravel(ytt)

#test data
test_df = pd.read_csv('test.csv', header=0)

# fitting base models
# models used: (bayesian) ridge regression, svm, rf, extra trees, gbr

# ridge regression
ridge_model = Ridge()
ridge_model.fit(xtr,ytr)
mean_squared_error(ridge_pred, ytt)

# bayesian ridge
b_ridge_model = BayesianRidge()
b_ridge_model.fit(xtr,ytr)
mean_squared_error(b_ridge_pred, ytt)

#svm
svm_model = SVR()
svm_model.fit(xtr, ytr)
mean_squared_error(svm_model.predict(xtt), ytt)

# RF
rf_model = RandomForestRegressor()
rf_model.fit(xtr, ytr)
mean_squared_error(rf_model.predict(xtt), ytt)

# extra trees
erf_model = ExtraTreesRegressor()
erf_model.fit(xtr, ytr)
mean_squared_error(erf_model.predict(xtt), ytt)

#gb
gb_model = GradientBoostingRegressor()
gb_model.fit(xtr, ytr)
mean_squared_error(gb_model.predict(xtt), ytt)

# from base models, gb, rf, svr best in descending order

# gb tuning hyperparameter
gb_1 = GradientBoostingRegressor(n_estimators=500, max_depth= 3, max_features=20)
gb_2 = GradientBoostingRegressor(n_estimators=300, max_depth= 3, max_features=20)
gb_3 = GradientBoostingRegressor(n_estimators=700, max_depth= 3, max_features=20)
gb_4 = GradientBoostingRegressor(n_estimators=150, max_depth= 3, max_features=20)

# gb_1 = GradientBoostingRegressor(n_estimators=500,max_depth= 3)
# gb_2 = GradientBoostingRegressor(n_estimators=300,max_depth= 3)
# gb_3 = GradientBoostingRegressor(n_estimators=700,max_depth= 3)
# gb_4 = GradientBoostingRegressor(n_estimators=150,max_depth= 3)

gb_1.fit(xtr, ytr)
gb_2.fit(xtr, ytr)
gb_3.fit(xtr, ytr)
gb_4.fit(xtr, ytr)
print(mean_squared_error(gb_1.predict(xtt), ytt), mean_squared_error(gb_2.predict(xtt), ytt),
mean_squared_error(gb_3.predict(xtt), ytt), mean_squared_error(gb_4.predict(xtt), ytt))

#gb_2 consistently gets better results but only by a bit
# lower # of estimators = better?

# rf tuning hyperparameter

rf_1 = RandomForestRegressor(n_estimators=500, max_depth= 3, max_features=20)
rf_2 = RandomForestRegressor(n_estimators=300, max_depth= 3, max_features=20)
rf_3 = RandomForestRegressor(n_estimators=700, max_depth= 3, max_features=20)
rf_1.fit(xtr, ytr)
rf_2.fit(xtr, ytr)
rf_3.fit(xtr, ytr)
print(mean_squared_error(rf_1.predict(xtt), ytt), mean_squared_error(rf_2.predict(xtt), ytt),
mean_squared_error(rf_3.predict(xtt), ytt))
