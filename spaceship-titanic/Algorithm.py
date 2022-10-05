import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

prep = Pipeline([
    ('std', StandardScaler()),
    ('nrm', Normalizer())
])
la = LabelEncoder()
def prepare_input(csv):
    klm = pd.read_csv(csv)
    pre = klm[['CryoSleep', 'VIP']].values
    cabin_side = klm['Cabin'].str.split('/').str[-1].values
    #cabin_deck = klm['Cabin'].str.split('/').str[0].values
    #cabin_num = klm['Cabin'].str.split('/').str[1].values
    #cabin = np.concatenate((cabin_deck[:, None], cabin_side[:,None]), axis = 1)
    trans = np.concatenate((pre, cabin_side[:, None]), axis = 1)
    for i in range(len(trans[0])):
        trans[:, i] = la.fit_transform(trans[:, i])
    merg = klm[['Age']].fillna(0).values
    x = np.concatenate((trans, merg), axis = 1)
    x = prep.fit_transform(x)
    return x
def prepare_output(csv):
    y = pd.read_csv(csv)[['Transported']].values
    y = la.fit_transform(y.ravel())
    return y
def fit_func(x, y):
    clf  = LogisticRegression(random_state = 0)
    clf.fit(x, y.ravel())
    return clf   

x = prepare_input('train.csv')
y = prepare_output('train.csv')
x_test = prepare_input('test.csv')
y_test = prepare_output('sample_submission.csv')

#x_train, x_t, y_train, y_t = train_test_split(x, y, test_size = 0.2, random_state = 50)
model = fit_func(x, y)
pred = model.predict(x)
score = accuracy_score(y, pred)
print('train score : ', score)
pred = model.predict(x_test)
score = accuracy_score(y_test, pred)
print('test score : ', score)
