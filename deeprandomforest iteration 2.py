from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV

        
x_train = pd.read_csv('influencer_train_log.csv', header = 0)
y_train = pd.read_csv('influencer_label.csv', header = 0)
x_test = pd.read_csv('influencer_test_log.csv', header = 0)

# first hidden layer
clf_1 = RandomForestClassifier(n_estimators=500)
clf_1.fit(x_train, np.ravel(y_train))
train_pred_1 = clf_1.predict_proba(x_train)
test_pred_1 = clf_1.predict_proba(x_test)
clf_2 = RandomForestClassifier(n_estimators=300)
clf_2.fit(x_train, np.ravel(y_train))
train_pred_2 = clf_2.predict_proba(x_train)
test_pred_2 = clf_2.predict_proba(x_test)
clf_3 = ExtraTreesClassifier(n_estimators=500)
clf_3.fit(x_train, np.ravel(y_train))
train_pred_3 = clf_3.predict_proba(x_train)
test_pred_3 = clf_3.predict_proba(x_test)
clf_4 = ExtraTreesClassifier(n_estimators=300)
clf_4.fit(x_train, np.ravel(y_train))
train_pred_4 = clf_4.predict_proba(x_train)
test_pred_4 = clf_4.predict_proba(x_test)

new_train_features = np.c_[train_pred_1, train_pred_2, train_pred_3, train_pred_4, x_train]
new_test_features = np.c_[test_pred_1, test_pred_3 ,test_pred_3, test_pred_4, x_test]

# 2nd hidden layer
clf_1 = RandomForestClassifier(n_estimators=500)
clf_1.fit(new_train_features, np.ravel(y_train))
train_pred_1 = clf_1.predict_proba(new_train_features)
test_pred_1 = clf_1.predict_proba(new_test_features)
clf_2 = RandomForestClassifier(n_estimators=300)
clf_2.fit(new_train_features, np.ravel(y_train))
train_pred_2 = clf_2.predict_proba(new_train_features)
test_pred_2 = clf_2.predict_proba(new_test_features)
clf_3 = ExtraTreesClassifier(n_estimators=500)
clf_3.fit(new_train_features, np.ravel(y_train))
train_pred_3 = clf_3.predict_proba(new_train_features)
test_pred_3 = clf_3.predict_proba(new_test_features)
clf_4 = ExtraTreesClassifier(n_estimators=300)
clf_4.fit(new_train_features, np.ravel(y_train))
train_pred_4 = clf_4.predict_proba(new_train_features)
test_pred_4 = clf_4.predict_proba(new_test_features)

new_train_features = np.c_[train_pred_1, train_pred_2, train_pred_3, train_pred_4, x_train]
new_test_features = np.c_[test_pred_1, test_pred_3 ,test_pred_3, test_pred_4, x_test]

# 3d hidden layer
clf_1 = RandomForestClassifier(n_estimators=500)
clf_1.fit(new_train_features, np.ravel(y_train))
train_pred_1 = clf_1.predict_proba(new_train_features)
test_pred_1 = clf_1.predict_proba(new_test_features)
clf_2 = RandomForestClassifier(n_estimators=300)
clf_2.fit(new_train_features, np.ravel(y_train))
train_pred_2 = clf_2.predict_proba(new_train_features)
test_pred_2 = clf_2.predict_proba(new_test_features)
clf_3 = ExtraTreesClassifier(n_estimators=500)
clf_3.fit(new_train_features, np.ravel(y_train))
train_pred_3 = clf_3.predict_proba(new_train_features)
test_pred_3 = clf_3.predict_proba(new_test_features)
clf_4 = ExtraTreesClassifier(n_estimators=300)
clf_4.fit(new_train_features, np.ravel(y_train))
train_pred_4 = clf_4.predict_proba(new_train_features)
test_pred_4 = clf_4.predict_proba(new_test_features)

new_train_features = np.c_[train_pred_1, train_pred_2, train_pred_3, train_pred_4, x_train]
new_test_features = np.c_[test_pred_1, test_pred_3 ,test_pred_3, test_pred_4, x_test]

# 4th hidden layer
clf_1 = RandomForestClassifier(n_estimators=500)
clf_1.fit(new_train_features, np.ravel(y_train))
train_pred_1 = clf_1.predict_proba(new_train_features)
test_pred_1 = clf_1.predict_proba(new_test_features)
clf_2 = RandomForestClassifier(n_estimators=300)
clf_2.fit(new_train_features, np.ravel(y_train))
train_pred_2 = clf_2.predict_proba(new_train_features)
test_pred_2 = clf_2.predict_proba(new_test_features)
clf_3 = ExtraTreesClassifier(n_estimators=500)
clf_3.fit(new_train_features, np.ravel(y_train))
train_pred_3 = clf_3.predict_proba(new_train_features)
test_pred_3 = clf_3.predict_proba(new_test_features)
clf_4 = ExtraTreesClassifier(n_estimators=300)
clf_4.fit(new_train_features, np.ravel(y_train))
train_pred_4 = clf_4.predict_proba(new_train_features)
test_pred_4 = clf_4.predict_proba(new_test_features)

new_train_features = np.c_[train_pred_1, train_pred_2, train_pred_3, train_pred_4, x_train]
new_test_features = np.c_[test_pred_1, test_pred_3 ,test_pred_3, test_pred_4, x_test]

#output layer
clf_1 = RandomForestClassifier(n_estimators=500)
clf_1.fit(new_train_features, np.ravel(y_train))
#train_pred_1 = clf_1.predict_proba(new_train_features)
test_pred_1 = clf_1.predict_proba(new_test_features)
clf_2 = RandomForestClassifier(n_estimators=300)
clf_2.fit(new_train_features, np.ravel(y_train))
#train_pred_2 = clf_2.predict_proba(new_train_features)
test_pred_2 = clf_2.predict_proba(new_test_features)
clf_3 = ExtraTreesClassifier(n_estimators=500)
clf_3.fit(new_train_features, np.ravel(y_train))
#train_pred_3 = clf_3.predict_proba(new_train_features)
test_pred_3 = clf_3.predict_proba(new_test_features)
clf_4 = ExtraTreesClassifier(n_estimators=300)
clf_4.fit(new_train_features, np.ravel(y_train))
#train_pred_4 = clf_4.predict_proba(new_train_features)
test_pred_4 = clf_4.predict_proba(new_test_features)
drf_pred = (test_pred_1+ test_pred_2+ test_pred_3+test_pred_4)/4
drf_pred = drf_pred[:,1:2]

#gradient boosting
gb_optimized_clf = GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='exponential', max_depth=15,
              max_features='log2', max_leaf_nodes=None,
              min_impurity_split=1e-07, min_samples_leaf=1,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              n_estimators=500, presort='auto', random_state=None,
              subsample=0.1, verbose=0, warm_start=False)
              
gb_optimized_clf.fit(x_train, np.ravel(y_train)) #optimized gradient boosted trees
gb_pred = gb_optimized_clf.predict_proba(x_test)
gb_pred = gb_pred[:,1:2]

prediction = 0*drf_pred + gb_pred
prediction.tofile('test sub.csv',sep=',',format='%10.5f')

#print(np.average(cross_val_score(clf_1, new_train_features, np.ravel(y_train), cv = 5)))
#print(np.average(cross_val_score(clf_2, new_train_features, np.ravel(y_train), cv = 5)))
#print(np.average(cross_val_score(clf_3, new_train_features, np.ravel(y_train), cv = 5)))



#0.81279 private 0.81510 public

# gcforest = 0.81494, 0.83306 (after playing with structure)