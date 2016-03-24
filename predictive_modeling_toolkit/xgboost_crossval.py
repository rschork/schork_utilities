import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.cross_validation import train_test_split

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

def modelfit(alg, dtrain, predictors, dtest=None, dscore=None, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['logloss'], early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['target'], eval_metric='logloss')

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    if isinstance(dtest, pd.DataFrame):
        dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]
    if isinstance(dscore, pd.DataFrame):
        dscore_predprob = alg.predict_proba(dscore[predictors])[:,1]
        np.savetxt('XGBoost_pred_raw.csv', dscore_predprob, delimiter=",")

    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['target'].values, dtrain_predictions)
    print "Metric Score (Train): %f" % metrics.log_loss(dtrain['target'], dtrain_predprob)
    if isinstance(dtest, pd.DataFrame):
        print "Metric Score (Test): %f" % metrics.log_loss(dtest['target'], dtest_predprob)

    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()

if __name__ == '__main__':

    train = pd.read_csv('/Users/ryanschork/schork_utilities/numerai_datasets/numerai_training_data.csv')
    score = pd.read_csv('/Users/ryanschork/schork_utilities/numerai_datasets/numerai_tournament_data.csv')
    target = 'target'
    IDcol = ''

    predictors = [x for x in train.columns if x not in [target, IDcol]]

    # X = train[predictors]
    # y = pd.DataFrame(train['target'])
    #
    # train, test, y_train, y_test = train_test_split(X, y, test_size=.15, stratify=y)
    #
    # train['target'] = y_train
    # test['target'] = y_test

    #Choose all predictors except target & IDcols

    # xgb1 = XGBClassifier(
    #  learning_rate =0.1,
    #  n_estimators=1000,
    #  max_depth=5,
    #  min_child_weight=1,
    #  gamma=0,
    #  subsample=0.8,
    #  colsample_bytree=0.8,
    #  objective= 'binary:logistic',
    #  nthread=4,
    #  scale_pos_weight=1,
    #  seed=27)
    # modelfit(xgb1, train, predictors, dtest=test)

    # param_test1 = {
    #  'max_depth':range(3,10,2),
    #  'min_child_weight':range(1,6,2)
    # }
    # gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=14, max_depth=5,
    #  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),
    #  param_grid = param_test1, scoring='log_loss',n_jobs=4,iid=False, cv=5)
    # gsearch1.fit(train[predictors],train[target])
    # print gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
    #
    # param_test2 = {
    #  'max_depth':[4,5,6],
    #  'min_child_weight':[4,5,6]
    # }
    # gsearch2 = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=14, max_depth=5,
    #  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
    #  param_grid = param_test2, scoring='log_loss',n_jobs=4,iid=False, cv=5)
    # gsearch2.fit(train[predictors],train[target])
    # print gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
    #
    # param_test2b = {
    #  'min_child_weight':[6,8,10,12]
    # }
    # gsearch2b = GridSearchCV(estimator = XGBClassifier( learning_rate=0.1, n_estimators=14, max_depth=4,
    #  min_child_weight=2, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
    #  param_grid = param_test2b, scoring='log_loss',n_jobs=4, iid=False, cv=5)
    # gsearch2b.fit(train[predictors],train[target])
    #
    # modelfit(gsearch2b.best_estimator_, train, predictors, dtest=test)
    # print gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_
    #
    # param_test3 = {
    #  'gamma':[i/10.0 for i in range(0,5)]
    # }
    # gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=14, max_depth=4,
    #  min_child_weight=8, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
    #  param_grid = param_test3, scoring='log_loss', n_jobs=4, iid=False, cv=5)
    # gsearch3.fit(train[predictors],train[target])
    # print gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
    #
    # xgb2 = XGBClassifier(
    #  learning_rate =0.1,
    #  n_estimators=1000,
    #  max_depth=4,
    #  min_child_weight=8,
    #  gamma=0,
    #  subsample=0.8,
    #  colsample_bytree=0.8,
    #  objective= 'binary:logistic',
    #  nthread=4,
    #  scale_pos_weight=1,
    #  seed=27)
    # modelfit(xgb2, train, predictors, dtest=test)
    #
    # param_test4 = {
    #  'subsample':[i/10.0 for i in range(6,10)],
    #  'colsample_bytree':[i/10.0 for i in range(6,10)]
    # }
    # gsearch4 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=14, max_depth=4,
    #  min_child_weight=8, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
    #  param_grid = param_test4, scoring='log_loss',n_jobs=4,iid=False, cv=5)
    # gsearch4.fit(train[predictors],train[target])
    # print gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
    #
    # param_test5 = {
    #  'subsample':[i/100.0 for i in range(55,90,5)],
    #  'colsample_bytree':[i/100.0 for i in range(75,90,5)]
    # }
    # gsearch5 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=16, max_depth=4,
    #  min_child_weight=8, gamma=0, subsample=0.8, colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
    #  param_grid = param_test5, scoring='log_loss',n_jobs=4,iid=False, cv=5)
    # gsearch5.fit(train[predictors],train[target])
    # print gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_
    #
    # param_test6 = {
    #  'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
    # }
    # gsearch6 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=16, max_depth=4,
    #  min_child_weight=8, gamma=0.1, subsample=0.7, colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
    #  param_grid = param_test6, scoring='log_loss',n_jobs=4,iid=False, cv=5)
    # gsearch6.fit(train[predictors],train[target])
    # print gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

    # param_test7 = {
    #  'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
    # }
    # gsearch7 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=16, max_depth=4,
    #  min_child_weight=8, gamma=0.1, subsample=0.7, colsample_bytree=0.8,
    #  objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27),
    #  param_grid = param_test7, scoring='log_loss',n_jobs=4,iid=False, cv=5)
    # gsearch7.fit(train[predictors],train[target])
    # print gsearch7.grid_scores_, gsearch7.best_params_, gsearch7.best_score_
    #
    # xgb3 = XGBClassifier(
    #  learning_rate =0.1,
    #  n_estimators=1000,
    #  max_depth=4,
    #  min_child_weight=8,
    #  gamma=0,
    #  subsample=0.7,
    #  colsample_bytree=0.8,
    #  reg_alpha=0.005,
    #  objective= 'binary:logistic',
    #  nthread=4,
    #  scale_pos_weight=1,
    #  seed=27)
    # modelfit(xgb3, train, predictors, dtest=test)

    xgb4 = XGBClassifier(
     learning_rate =0.01,
     n_estimators=5000,
     max_depth=4,
     min_child_weight=8,
     gamma=0,
     subsample=0.7,
     colsample_bytree=0.8,
     reg_alpha=0.005,
     objective= 'binary:logistic',
     nthread=4,
     scale_pos_weight=1,
     seed=27)
    modelfit(xgb4, train, predictors, dscore=score)
