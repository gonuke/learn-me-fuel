#! /usr/bin/env python

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

def errors_and_scores(trainX, Y, knn_init, rr_init, svr_init, rxtr_pred, scores, CV):
    """
    Saves csv's with each reactor parameter regression wrt scoring metric and 
    algorithm

    """
    cols = ['r2 Score', 'Explained Variance', 'Negative MAE', 'Negative RMSE']
    idx = ['kNN', 'Ridge', 'SVR']
    # init/empty the lists
    knn_scores = []
    rr_scores = []
    svr_scores = []
    for alg in ('knn', 'rr', 'svr'):
        # get init'd learner
        if alg == 'knn':
            alg_pred = knn_init
        elif alg == 'rr':
            alg_pred = rr_init
        else:
            alg_pred = svr_init
        
        #r2, exp_var, mae, mse = cross_validate(alg_pred, trainX, Y, scoring=scores, cv=CV)
        #rmse =-1 * np.sqrt(-1*mse)
        cv_info = cross_validate(alg_pred, trainX, Y, scoring=scores, cv=CV)
        df = pd.DataFrame(cv_info)
        train_mse = df['train_neg_mean_squared_error']
        test_mse = df['test_neg_mean_squared_error']
        rmse_calc = lambda x, y : -1 * np.sqrt(-1*x)
        df['train_neg_rmse'] = rmse_calc 
        df['test_neg_rmse'] = rmse_calc
        

        #score_nums = [r2, exp_var, mae, rmse]
        if alg == 'knn':
            knn_scores = score_nums
        elif alg == 'rr':
            rr_scores = score_nums
        else:
            svr_scores = score_nums
    df = pd.DataFrame([knn_scores, rr_scores, svr_scores], index=idx, columns=cols)
    df.to_csv('sfcompo_' + rxtr_pred + '_scores.csv')
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

    pkl_name = './sfcompo_pickles/not-scaled_trainset_nucs_fissact_8dec.pkl'
    trainXY = pd.read_pickle(pkl_name)
    trainX, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX)
    
    CV = 5
    scores = ['r2', 'explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error']
    # The hand-picked numbers are based on the dayman test set validation curves
    k = 13
    a = 100
    g = 0.001
    c = 10000
    # loops through each reactor parameter to do separate predictions
    for Y in ('c', 'e', 'b'):
        trainY = pd.Series()
        if Y == 'c':
            trainY = cY
            parameter = 'cooling'
        elif Y == 'e': 
            trainY = eY
            parameter = 'enrichment'
        else:
            trainY = bY
            parameter = 'burnup'
        # initialize a learner
        knn_init = KNeighborsRegressor(n_neighbors=k)
        rr_init = Ridge(alpha=a)
        svr_init = SVR(gamma=g, C=c)
        # make predictions
        #knn = cross_val_predict(knn_init, trainX, y=trainY, cv=CV)
        #rr = cross_val_predict(rr_init, trainX, y=trainY, cv=CV)
        #svr = cross_val_predict(svr_init, trainX, y=trainY, cv=CV)
        #preds_by_alg = pd.DataFrame({'TrueY': trainY, 'kNN': knn, 
        #                             'Ridge': rr, 'SVR': svr}, 
        #                            index=trainY.index)
        #preds_by_alg.to_csv('sfcompo_' + parameter + '_predictions.csv')
        # calculate errors and scores
        errors_and_scores(trainX, trainY, knn_init, rr_init, svr_init, parameter, scores, CV)
    return

if __name__ == "__main__":
    main()
