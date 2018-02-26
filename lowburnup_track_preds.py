#! /usr/bin/env/ python

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

def errors_and_scores(testY, knn, rr, svr, rxtr_pred):
    """
    Saves csv's with each reactor parameter regression wrt scoring metric and 
    algorithm

    """
    cols = ['r2 Score', 'Explained Variance', 'Negative MAE', 'Negative RMSE']
    idx = ['kNN', 'Ridge', 'SVR']
    for alg_pred in (knn, rr, svr):
        # 4 calculations of various 'scores':
        r2 = r2_score(testY, alg_pred)
        exp_var = explained_variance(testY, alg_pred)
        mae = -1 * mean_absolute_error(testY, alg_pred)
        rmse =-1 * np.sqrt(mean_squared_error(testY, alg_pred))
        scores = [r2, exp_var, mae, rmse]
        # init/empty the lists
        knn_scores = []
        rr_scores = []
        svr_scores = []
        if alg_pred == knn:
            knn_scores = scores
        elif alg_pred == rr:
            rr_scores = scores
        else:
            svr_scores = scores
    df = pd.DataFrame([knn_scores, rr_scores, svr_scores], index=idx, columns=cols)
    df.to_csv('lowburn_' + rxtr_pred + '_scores.csv')
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
    Given training data, this script trains and tracks each prediction for
    several algorithms and saves the predictions and ground truth to a CSV file
    """

    pkl_train = './lowburnup_pickles/trainXY_2nov.pkl'
    trainXY = pd.read_pickle(pkl_name, compression=None)
    trainX, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX)
    pkl_test = './lowburnup_pickles/testXY_2nov.pkl'
    testXY = pd.read_pickle(pkl_name, compression=None)
    testX, test_rY, test_cY, test_eY, test_bY = splitXY(testXY)
    
    # Add some auto-optimize-param stuff here but it's a constant for now
    # The hand-picked numbers are based on the dayman test set validation curves
    k = 13
    a = 1000
    g = 0.001
    c = 1000
    # loops through each reactor parameter to do separate predictions
    for trainY in (cY, eY, bY):
        testY = pd.DataFrame()
        if Y == cY:
            parameter = 'cooling'
            testY = test_cY
        elif Y == eY:
            parameter = 'enrichment'
            testY = test_eY
        else:
            parameter = 'burnup'
            testY = test_bY
        # initialize a learner
        knn_init = KNeighborsRegressor(n_neighbors=k)
        rr_init = Ridge(alpha=a)
        svr_init = SVR(gamma=g, C=c)
        # fit w data
        knn_init.fit(trainX, trainY)
        rr_init.fit(trainX, trainY)
        svr_init.fit(trainX, trainY)
        # make predictions
        knn = knn_init.predict(testX)
        rr = rr_init.predict(testX)
        svr = svr_init.predict(testX)
        preds_by_alg = pd.DataFrame({'TrueY': testY, 'kNN': knn, 
                                     'Ridge': rr, 'SVR': svr}, 
                                    index=testY.index)
        preds_by_alg.to_csv('lowburn_' + parameter + '_predictions.csv')
        # calculate errors and scores
        #errors_and_scores(testY, knn, rr, svr, parameter)
    return

if __name__ == "__main__":
    main()
