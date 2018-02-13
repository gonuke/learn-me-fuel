#! /usr/bin/env/ python

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import scale
import pandas as pd

def train_preds(trainX, trainY, testX, testY, knn_init, rr_init, svr_init, rxtr_pred):
    """
    Saves csv's with each regression option wrt scoring metric and algorithm

    """
    # Set cross-validation folds
    CV = 10

    scores = ['r2', 'explained_variance_score', 
              'neg_mean_absolute_error', 'neg_mean_squared_error'
              ]

    # fit w data
    knn_init.fit(trainX, trainY)
    rr_init.fit(trainX, trainY)
    svr_init.fit(trainX, trainY)
    # initialize Series with tracking the testing instances for plotting purposes 
    knn_scores = testY
    rr_scores = testY
    svr_scores = testY
    for score in scores:
        # Set cross-validation folds
        CV = 10

        # kNN
        # need to check if this predict round can be done before or after saying the type of score, i.e., inside or outside of loop
        # this predict round needs to track the entire testing set's pred errors
        # cv predict doesn't take a score, might have to manually calc (via error_of_choice(testY, cv_pred))
        # any shortcut for fit -> cv pred of textX? -> manual error calc?
        knn = cross_val_predict(knn_init, trainX, trainY, cv=CV, scoring=score) 
        knn_score = pd.Series(knn, index=testY.index, name= rxtr_pred + ' knn ' + score)
        knn_scores = pd.concat(knn_score, axis=1)
        # Ridge
        rr = cross_val_predict(rr_init, trainX, trainY, cv=CV, scoring=score) 
        rr_score = pd.Series(rr, index=testY.index, name= rxtr_pred + ' rr ' + score)
        rr_scores = pd.concat(rr_score, axis=1)
        # SVR
        svr = cross_val_predict(svr_init, trainX, trainY, cv=CV, scoring=score)
        svr_score = pd.Series(svr, index=testY.index, name= rxtr_pred + ' svr ' + score)
        svr_scores = pd.concat(svr_score, axis=1)
    # save dataframe with scores/errors to CSV
    knn_scores.to_csv('knn_' + rxtr_pred + '.csv')
    rr_scores.to_csv('rr_' + rxtr_pred + '.csv')
    svr_scores.to_csv('svr_' + rxtr_pred + '.csv')

    return

def splitXY(dfXY):
    """
    Takes a dataframe with all X (features) and Y (labels) information and 
    produces five different pandas datatypes: a dataframe with nuclide info 
    only + a series for each label column.

    Parameters
    ----------
    dfXY : dataframe with nuclide concentraations and 4 labels: reactor type, 
           cooling time, enrichment, and burnup

    Returns
    -------
    dfX : dataframe with only nuclide concentrations for each instance
    rY : dataframe with reactor type for each instance
    cY : dataframe with cooling time for each instance
    eY : dataframe with fuel enrichment for each instance
    bY : dataframe with fuel burnup for each instance

    """

    lbls = ['ReactorType', 'CoolingTime', 'Enrichment', 'Burnup', 'total']
    dfX = dfXY.drop(lbls, axis=1)
    r_dfY = dfXY.loc[:, lbls[0]]
    c_dfY = dfXY.loc[:, lbls[1]]
    e_dfY = dfXY.loc[:, lbls[2]]
    b_dfY = dfXY.loc[:, lbls[3]]
    return dfX, r_dfY, c_dfY, e_dfY, b_dfY


def main():
    """
    Given training data, this script trains and tracks each prediction 

    Parameters 
    ---------- 
    
    train : group of dataframes that include training data and the three
            labels 
    
    Returns
    -------
    burnup : tuples of error metrics for training, testing, and cross validation 
             errors for all three algorithms

    """

    pkl_train = 'trainXY_2nov.pkl'
    trainXY = pd.read_pickle(pkl_name, compression=None)
    trainX, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX)
    # getting scores from test set now, not training set 
    pkl_test = 'testXY_2nov.pkl'
    testXY = pd.read_pickle(pkl_name, compression=None)
    testX, test_rY, test_cY, test_eY, test_bY = splitXY(testXY)
    
    # Add some auto-optimize-param stuff here but it's a constant for now
    # The hand-picked numbers are based on the dayman test set validation curves
    k = 13
    a = 1000
    g = 0.001
    c = 1000

    for trainY in (cY, eY, bY):
        if trainY == cY:
            parameter = 'cooling'
            testY = test_cY
        elif trainY == eY:
            parameter = 'enrichment'
            testY = test_eY
        else:
            parameter = 'burnup'
            testY = test_bY

        knn_init = KNeighborsRegressor(n_neighbors=k)
        rr_init = Ridge(alpha=a)
        svr_init = SVR(gamma=g, C=c)
        train_preds(trainX, trainY, testX, testY, knn_init, rr_init, svr_init, parameter)
        
    return

if __name__ == "__main__":
    main()
