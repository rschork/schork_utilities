import math
import matplotlib.pyplot as plt
import csv

from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn import __version__ as sklearn_version
from sklearn.svm import SVR
from pandas import *
from numpy import *
from sklearn.neighbors import KNeighborsRegressor

model_pre = DataFrame.from_csv('ice_cream_data.csv',index_col=False)
print model_pre
target = model_pre['sales'].astype(float)
feature_ar = model_pre['weekday']
feature = model_pre['temp'].astype(float)
feature_norm = (feature - ((feature.max() - feature.min())/2)) / (feature.max() - ((feature.max() - feature.min())/2))

feature_norm_x = pandas.concat([feature_ar, feature_norm], axis=1)

X = feature_norm_x.as_matrix()
Y = target.as_matrix()

train_to = 28

cv_method = KFold(len(X[0:train_to]), 10)
gamma_range = [0.0001, 0.000000001]
C_range = [2 ** i for i in range(1, 10, 1)]

tuned_parameters = [{
   'kernel': ['rbf'],
   'C': C_range,
   'gamma': gamma_range}]

# search for the best parameters with crossvalidation.
print 'searching...'

grid = GridSearchCV(SVR(kernel='rbf', epsilon = 0.05),\
   param_grid = tuned_parameters, cv=cv_method, verbose = 1)

print X[0:train_to]
print Y[0:train_to]
gridfit = grid.fit(X[0:train_to], Y[0:train_to])

print 'printing GRID', grid
print 'printing GRIDFIT', gridfit
print 'printing BEST PARAMS', grid.best_params_
print 'printing GRID BEST EST', grid.best_estimator_
print 'PRINT BEST SCORE', grid.best_score_

# train a SVR regressor with best found parameters.
svr = SVR(kernel='rbf', epsilon=0.05, C = grid.best_estimator_.C,\
   gamma = grid.best_estimator_.gamma)
svr.fit(X[0:train_to], Y[0:train_to])
y_hat = svr.predict(X[train_to::])

print 'prediction! ->', y_hat
