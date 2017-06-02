import math as m
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score

#os.chdir("C:\\Users\\jc_ch\\Desktop\\ML Stuff")

## For one set of data, either sequence or matrix, generate multi grained scanned results
## This method is defined but unused in the current context of the project
def mlt_gr_scan (feat, struct, size, size2):
    #res = [[] for l in range(2)]
    if (struct != 'seq' and struct != 'mat'):
        return 0
    if (struct == 'seq'):
        rep = len(feat) - size + 1
        feat_lst = [[] for i in range(rep)] 
        for i in range(0,rep):
            feat_lst[i] = feat[i:i+size]     
            #lab_lst [i] = lab []
    else:
        num_row = feat.shape[0]
        num_col = feat.shape[1]
        rep = num_row - size + 1
        rep_c = num_col - size2 + 1 
        feat_lst = [[] for i in range(rep*rep_c)]
        k = 0
        for i in range(0,rep):
            for j in range(0,rep_c):
                feat_lst[k] = feat [i:i+size,j:j+size2]  
                #lab_lst [k] = lab []
                k+=1
    #res[0] = feat_lst
    #res[1] = lab_lst
    return feat_lst

## For one dataset of features, generate multi grain scanned results using a 
## sequenced basis for the full dataset
def mltGrScCol (feat, lab, size):
    num_col = feat.shape[1]
    rep = num_col - size + 1
    feat_lst = [[] for i in range(rep)]
    for i in range(0,rep):
        feat_lst[i] = feat[:,i:i+size]     
    return [feat_lst, lab]

x_train = pd.read_csv('ret_train.csv', header = 0)
y_train = pd.read_csv('roro_train.csv', header = 0)
x_test = pd.read_csv('ret_test.csv', header = 0)
y_test = pd.read_csv('roro_test.csv', header = 0)

# win1 = 5
# win2 = 10
# win3 = 15

# For your daily purposes Professor Ali
win1 = m.floor(x_train.shape[1]/4)
win2 = m.floor(x_train.shape[1]/9)
win3 = m.floor(x_train.shape[1]/16)

xtr = x_train.as_matrix()
ytr = y_train.as_matrix()
xte = x_test.as_matrix()


training_new = [[] for i in range(3)]
testing_new = [[] for i in range(3)]

training_new[0] = mltGrScCol (xtr, ytr, win1)
testing_new[0] = mltGrScCol (xte, 0, win1)
training_new[1] = mltGrScCol (xtr, ytr, win2)
testing_new[1] = mltGrScCol (xte, 0, win2)
training_new[2] = mltGrScCol (xtr, ytr, win3)
testing_new[2] = mltGrScCol (xte, 0, win3)

iters = [[] for i in range(3)]
iters[0] = len(training_new[0][0])
iters[1] = len(training_new[1][0])
iters[2] = len(training_new[2][0])

ftr = [None]*1040
fte = [None]*260
for j in range(3):      
    for i in range(0,iters[j]):
        clf_1 = RandomForestClassifier(n_estimators=30)
        clf_1.fit(training_new[j][0][i], np.ravel(training_new[j][1]))
        train_pred_1 = clf_1.predict_proba(training_new[j][0][i])
        test_pred_1 = clf_1.predict_proba(testing_new[j][0][i])
        clf_2 = ExtraTreesClassifier(n_estimators=30)
        clf_2.fit(training_new[j][0][i], np.ravel(training_new[j][1]))
        train_pred_2 = clf_1.predict_proba(training_new[j][0][i])
        test_pred_2 = clf_1.predict_proba(testing_new[j][0][i])
        ftr = np.c_[ftr,train_pred_1,train_pred_2]
        fte = np.c_[fte,test_pred_1,test_pred_2]

ftr = np.delete(ftr, (0), axis=1)
fte = np.delete(fte, (0), axis=1)

# first hidden layer
clf_1 = RandomForestClassifier(n_estimators=700)
clf_1.fit(ftr, np.ravel(y_train))
train_pred_1 = clf_1.predict_proba(ftr)
test_pred_1 = clf_1.predict_proba(fte)
clf_2 = RandomForestClassifier(n_estimators=500)
clf_2.fit(ftr, np.ravel(y_train))
train_pred_2 = clf_2.predict_proba(ftr)
test_pred_2 = clf_2.predict_proba(fte)
clf_3 = ExtraTreesClassifier(n_estimators=700)
clf_3.fit(ftr, np.ravel(y_train))
train_pred_3 = clf_3.predict_proba(ftr)
test_pred_3 = clf_3.predict_proba(fte)
clf_4 = ExtraTreesClassifier(n_estimators=500)
clf_4.fit(ftr, np.ravel(y_train))
train_pred_4 = clf_4.predict_proba(ftr)
test_pred_4 = clf_4.predict_proba(fte)

new_train_features = np.c_[train_pred_1, train_pred_2, train_pred_3, train_pred_4, ftr]
new_test_features = np.c_[test_pred_1, test_pred_3 ,test_pred_3, test_pred_4, fte]

# # 2nd hidden layer
clf_1 = RandomForestClassifier(n_estimators=700)
clf_1.fit(new_train_features, np.ravel(y_train))
train_pred_1 = clf_1.predict_proba(new_train_features)
test_pred_1 = clf_1.predict_proba(new_test_features)
clf_2 = RandomForestClassifier(n_estimators=500)
clf_2.fit(new_train_features, np.ravel(y_train))
train_pred_2 = clf_2.predict_proba(new_train_features)
test_pred_2 = clf_2.predict_proba(new_test_features)
clf_3 = ExtraTreesClassifier(n_estimators=700)
clf_3.fit(new_train_features, np.ravel(y_train))
train_pred_3 = clf_3.predict_proba(new_train_features)
test_pred_3 = clf_3.predict_proba(new_test_features)
clf_4 = ExtraTreesClassifier(n_estimators=500)
clf_4.fit(new_train_features, np.ravel(y_train))
train_pred_4 = clf_4.predict_proba(new_train_features)
test_pred_4 = clf_4.predict_proba(new_test_features)

new_train_features = np.c_[train_pred_1, train_pred_2, train_pred_3, train_pred_4, ftr]
new_test_features = np.c_[test_pred_1, test_pred_3 ,test_pred_3, test_pred_4, fte]

# # 3d hidden layer
clf_1 = RandomForestClassifier(n_estimators=700)
clf_1.fit(new_train_features, np.ravel(y_train))
train_pred_1 = clf_1.predict_proba(new_train_features)
test_pred_1 = clf_1.predict_proba(new_test_features)
clf_2 = RandomForestClassifier(n_estimators=500)
clf_2.fit(new_train_features, np.ravel(y_train))
train_pred_2 = clf_2.predict_proba(new_train_features)
test_pred_2 = clf_2.predict_proba(new_test_features)
clf_3 = ExtraTreesClassifier(n_estimators=700)
clf_3.fit(new_train_features, np.ravel(y_train))
train_pred_3 = clf_3.predict_proba(new_train_features)
test_pred_3 = clf_3.predict_proba(new_test_features)
clf_4 = ExtraTreesClassifier(n_estimators=500)
clf_4.fit(new_train_features, np.ravel(y_train))
train_pred_4 = clf_4.predict_proba(new_train_features)
test_pred_4 = clf_4.predict_proba(new_test_features)

new_train_features = np.c_[train_pred_1, train_pred_2, train_pred_3, train_pred_4, ftr]
new_test_features = np.c_[test_pred_1, test_pred_3 ,test_pred_3, test_pred_4, fte]

# 4th hidden layer
clf_1 = RandomForestClassifier(n_estimators=700)
clf_1.fit(new_train_features, np.ravel(y_train))
train_pred_1 = clf_1.predict_proba(new_train_features)
test_pred_1 = clf_1.predict_proba(new_test_features)
clf_2 = RandomForestClassifier(n_estimators=500)
clf_2.fit(new_train_features, np.ravel(y_train))
train_pred_2 = clf_2.predict_proba(new_train_features)
test_pred_2 = clf_2.predict_proba(new_test_features)
clf_3 = ExtraTreesClassifier(n_estimators=700)
clf_3.fit(new_train_features, np.ravel(y_train))
train_pred_3 = clf_3.predict_proba(new_train_features)
test_pred_3 = clf_3.predict_proba(new_test_features)
clf_4 = ExtraTreesClassifier(n_estimators=500)
clf_4.fit(new_train_features, np.ravel(y_train))
train_pred_4 = clf_4.predict_proba(new_train_features)
test_pred_4 = clf_4.predict_proba(new_test_features)

new_train_features = np.c_[train_pred_1, train_pred_2, train_pred_3, train_pred_4, ftr]
new_test_features = np.c_[test_pred_1, test_pred_3 ,test_pred_3, test_pred_4, fte]

# 5th hidden layer
clf_1 = RandomForestClassifier(n_estimators=700)
clf_1.fit(new_train_features, np.ravel(y_train))
train_pred_1 = clf_1.predict_proba(new_train_features)
test_pred_1 = clf_1.predict_proba(new_test_features)
clf_2 = RandomForestClassifier(n_estimators=500)
clf_2.fit(new_train_features, np.ravel(y_train))
train_pred_2 = clf_2.predict_proba(new_train_features)
test_pred_2 = clf_2.predict_proba(new_test_features)
clf_3 = ExtraTreesClassifier(n_estimators=700)
clf_3.fit(new_train_features, np.ravel(y_train))
train_pred_3 = clf_3.predict_proba(new_train_features)
test_pred_3 = clf_3.predict_proba(new_test_features)
clf_4 = ExtraTreesClassifier(n_estimators=500)
clf_4.fit(new_train_features, np.ravel(y_train))
train_pred_4 = clf_4.predict_proba(new_train_features)
test_pred_4 = clf_4.predict_proba(new_test_features)

new_train_features = np.c_[train_pred_1, train_pred_2, train_pred_3, train_pred_4, ftr]
new_test_features = np.c_[test_pred_1, test_pred_3 ,test_pred_3, test_pred_4, fte]


#output layer
clf_1 = RandomForestClassifier(n_estimators=700)
clf_1.fit(new_train_features, np.ravel(y_train))
#train_pred_1 = clf_1.predict_proba(new_train_features)
test_pred_1 = clf_1.predict_proba(new_test_features)
clf_2 = RandomForestClassifier(n_estimators=500)
clf_2.fit(new_train_features, np.ravel(y_train))
#train_pred_2 = clf_2.predict_proba(new_train_features)
test_pred_2 = clf_2.predict_proba(new_test_features)
clf_3 = ExtraTreesClassifier(n_estimators=700)
clf_3.fit(new_train_features, np.ravel(y_train))
#train_pred_3 = clf_3.predict_proba(new_train_features)
test_pred_3 = clf_3.predict_proba(new_test_features)
clf_4 = ExtraTreesClassifier(n_estimators=500)
clf_4.fit(new_train_features, np.ravel(y_train))
#train_pred_4 = clf_4.predict_proba(new_train_features)
test_pred_4 = clf_4.predict_proba(new_test_features)


drf_pred = (test_pred_1+ test_pred_2+ test_pred_3+test_pred_4)/4
drf_pred = drf_pred[:,1:2]
drf_pred = np.round(drf_pred)

accuracy_score(y_test, drf_pred)
