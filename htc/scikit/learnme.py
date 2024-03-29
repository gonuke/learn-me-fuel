#! /usr/bin/env python3

from tools import splitXY, track_predictions, errors_and_scores, validation_curves, learning_curves, ext_test_compare

from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, StratifiedKFold

import pandas as pd
import argparse

def main():
    """
    Given training data, this script performs a user-defined number of ML 
    tasks for three algorithms (kNN, decision trees, SVR), saving the results
    as .csv files:
    1. track_predictions provides the prediction of each training instance
    2. errors_and_scores provides the prediction accuracy for each algorithm
    for each CV fold
    3. learning_curves provides the prediction accuracy with respect to
    training set size
    4. validation_curves provides the prediction accuracy with respect to
    algorithm hyperparameter variations
    5. ext_test_compare provides the predictions of each algorithm of an 
    external test set

    """
    CV = 5
    tset_frac = 0.6
    
    parser = argparse.ArgumentParser(description='Performs machine learning-based predictions or model selection techniques.')
    parser.add_argument('rxtr_param', choices=['reactor', 'cooling', 'enrichment', 'burnup'], 
                        metavar='prediction-param', help='which reactor parameter is to be predicted [reactor, cooling, enrichment, burnup]')
    parser.add_argument('-tp', '--track_preds', action='store_true', 
                        default=False, help='run the track_predictions function')
    parser.add_argument('-es', '--err_n_scores', action='store_true', 
                        default=False, help='run the errors_and_scores function')
    parser.add_argument('-lc', '--learn_curves', action='store_true', 
                        default=False, help='run the learning_curves function')
    parser.add_argument('-vc', '--valid_curves', action='store_true', 
                        default=False, help='run the validation_curves function')
    #parser.add_argument('-tc', '--test_compare', action='store_true', 
    #                    default=False, help='run the ext_test_compare function')
    args = parser.parse_args()

    pkl = 'trainset.pkl'
    trainXY = pd.read_pickle(pkl)
    trainXY.reset_index(inplace=True, drop=True) 
    # hyperparam optimization was done on 60% of training set
    trainXY = trainXY.sample(frac=tset_frac)
    trainX_unscaled, rY, cY, eY, bY = splitXY(trainXY)
    trainX = scale(trainX_unscaled)
    
    csv_name = 'trainset_m60_' + args.rxtr_param
    
    # loops through each reactor parameter to do separate predictions
    # burnup is last since its the only tset I'm altering
    trainY = pd.Series()
    # get param names and set ground truth
    if args.rxtr_param == 'cooling':
        trainY = cY
        k = 3
        depth = 50
        feats = 25
        g = 0.06
        c = 50000
    elif args.rxtr_param == 'enrichment': 
        trainY = eY
        k = 7
        depth = 50
        feats = 25
        g = 0.8
        c = 25000
    elif args.rxtr_param == 'burnup':
        csv_name = 'trainset_m24_' + args.rxtr_param
        # burnup needs much less training data...this is 24% of data set
        trainXY = trainXY.sample(frac=0.4)
        trainX, rY, cY, eY, bY = splitXY(trainXY)
        trainX = scale(trainX)
        trainY = bY
        k = 7
        depth = 50
        feats = 25
        g = 0.25
        c = 42000
    else:
        trainY = rY
        k = 3
        depth = 50
        feats = 25 
        g = 0.07
        c = 1000
    
    ## initialize learners
    score = 'explained_variance'
    kfold = KFold(n_splits=CV, shuffle=True)
    knn_init = KNeighborsRegressor(n_neighbors=k, weights='distance')
    dtr_init = DecisionTreeRegressor(max_depth=depth, max_features=feats)
    svr_init = SVR(gamma=g, C=c)
    if args.rxtr_param == 'reactor':
        score = 'accuracy'
        kfold = StratifiedKFold(n_splits=CV, shuffle=True)
        knn_init = KNeighborsClassifier(n_neighbors=k, weights='distance')
        dtr_init = DecisionTreeClassifier(max_depth=depth, max_features=feats, class_weight='balanced')
        svr_init = SVC(gamma=g, C=c, class_weight='balanced')

    ## track predictions
    if args.track_preds == True:
        track_predictions(trainX, trainY, knn_init, dtr_init, svr_init, kfold, csv_name, trainX_unscaled)

    ## calculate errors and scores
    if args.err_n_scores == True:
        scores = ['r2', 'explained_variance', 'neg_mean_absolute_error']
        if args.rxtr_param == 'reactor':
            scores = ['accuracy', ]
        errors_and_scores(trainX, trainY, knn_init, dtr_init, svr_init, scores, kfold, csv_name)

    # learning curves
    if args.learn_curves == True:
        learning_curves(trainX, trainY, knn_init, dtr_init, svr_init, kfold, score, csv_name)
    
    # validation curves 
    if args.valid_curves == True:
        # VC needs different inits
        knn_init = KNeighborsRegressor(weights='distance')
        dtr_init = DecisionTreeRegressor()
        svr_init = SVR(C=c)
        if args.rxtr_param == 'reactor':
            knn_init = KNeighborsClassifier(weights='distance')
            dtr_init = DecisionTreeClassifier(class_weight='balanced')
            svr_init = SVC(C=c, class_weight='balanced')
        validation_curves(trainX, trainY, knn_init, dtr_init, svr_init, kfold, score, csv_name)
    
    # compare against external test set (right now the only one is 
    # Dayman test set)
    ##### 21 Jun 2019: this is untested and may not work without some of the 
    ##### algorithm imports added to tools.py
    #if args.test_compare == True:
    #    ext_test_compare(trainX, trainY, knn_init, dtr_init, svr_init, csv_name)

    return

if __name__ == "__main__":
    main()
