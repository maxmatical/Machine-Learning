import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('influencer_train.csv', header = 0)
df = np.asarray(df)
y = np.ravel(df[:,0])
x = df[:,1:23]
x_test = pd.read_csv('influencer_test.csv', header = 0)
np.asarray(x_test)

# starting with basic classifiers

#logistic regression
lr = LogisticRegression()
print(np.average(cross_val_score(lr, x, y, cv = 5)))

# KNN
knn = KNeighborsClassifier()
print(np.average(cross_val_score(knn, x, y, cv = 5)))

#random forest
rf = RandomForestClassifier()
print(np.average(cross_val_score(rf, x, y, cv = 5)))

# extra trees
erf = ExtraTreesClassifier()
print(np.average(cross_val_score(erf, x, y, cv = 5)))

# gradient boosting
gb = GradientBoostingClassifier()
print(np.average(cross_val_score(gb, x, y, cv = 5)))

#base case scenario gradiet boosting is best, then random forest

# generating extra sample points
from imblearn.over_sampling import SMOTE, ADASYN 
x_resampled, y_resampled = SMOTE().fit_sample(x,y)

# check
print(np.average(cross_val_score(lr, x_resampled, y_resampled, cv = 5)))
print(np.average(cross_val_score(knn, x_resampled, y_resampled, cv = 5)))
print(np.average(cross_val_score(rf, x_resampled, y_resampled, cv = 5)))
print(np.average(cross_val_score(erf, x_resampled, y_resampled, cv = 5)))
print(np.average(cross_val_score(gb, x_resampled, y_resampled, cv = 5)))
# resampled shows gb and rf still best

