
import ast
import json
import os
import pickle

import utils
import pandas as pd
import numpy as np

from scipy.stats import t


import sklearn as sk
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
import imblearn.under_sampling as resus
import imblearn.ensemble as resemb
import imblearn.combine as reshyb
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier


import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import seaborn as sns
try:
    plt.style.use('seaborn-whitegrid')
except:
    plt.style.use('seaborn-v0_8-whitegrid')

# import plotly as ply
# import plotly.express as px




# CONSTANT to store the random_state stated for reproducibility issues
RANDOM_STATE = 42

N_JOBS = 7

CV_N_SPLITS = 5


def sort_performances_results(df, cols_order_to_sort=['balanced_accuracy', 'sensitivity', 'specificity', 'fit_time'], cols_to_return=None):
    df_bests = df.sort_values(
        cols_order_to_sort, 
        ascending=[False, False, False, True]
    ).copy()
    if cols_to_return is not None:
        return df_bests[cols_to_return]
    else:
        return df_bests


def get_kfold_splits(n_splits=CV_N_SPLITS, random_state=RANDOM_STATE, shuffle_kfold=True, ):
    kfold = StratifiedKFold(
        n_splits=n_splits, 
        random_state = random_state if shuffle_kfold else None, 
        shuffle=shuffle_kfold,
    )
    return kfold



def get_default_scoring():
    # metrics to evaluate the model performance
    scoring = {
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'sensitivity': make_scorer(recall_score),
        'specificity': make_scorer(recall_score, pos_label=0),
        'f1': make_scorer(f1_score, zero_division=0.0),
        'auc': 'roc_auc', #make_scorer(roc_auc_score),
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0),
    }
    return scoring


def print_dataframe(filtered_cv_results):
    """Pretty print for filtered dataframe"""
    for mean_precision, std_precision, mean_recall, std_recall, params in zip(
        filtered_cv_results["mean_test_precision"],
        filtered_cv_results["std_test_precision"],
        filtered_cv_results["mean_test_recall"],
        filtered_cv_results["std_test_recall"],
        filtered_cv_results["params"],
    ):
        print(
            f"precision: {mean_precision:0.3f} (±{std_precision:0.03f}),"
            f" recall: {mean_recall:0.3f} (±{std_recall:0.03f}),"
            f" for {params}"
        )
    print()


 ## 
 # Define the strategy to select the best estimator.
 ##
def refit_strategy(cv_results):

    """The strategy defined here is to filter-out all results below a precision threshold
    of 0.98, rank the remaining by recall and keep all models with one standard
    deviation of the best by recall. Once these models are selected, we can select the
    fastest model to predict.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        CV results as returned by the `GridSearchCV`.

    Returns
    -------
    best_index : int
        The index of the best estimator as it appears in `cv_results`.
    """
    
    # print the info about the grid-search for the different scores

    precision_threshold = 0.98

    cv_results_ = pd.DataFrame(cv_results)
    print("All grid-search results:")
    print_dataframe(cv_results_)

    # Filter-out all results below the threshold
    high_precision_cv_results = cv_results_[
        cv_results_["mean_test_precision"] > precision_threshold
    ]

    print(f"Models with a precision higher than {precision_threshold}:")
    print_dataframe(high_precision_cv_results)

    high_precision_cv_results = high_precision_cv_results[
        [
            "mean_score_time",
            "mean_test_recall",
            "std_test_recall",
            "mean_test_precision",
            "std_test_precision",
            "rank_test_recall",
            "rank_test_precision",
            "params",
        ]
    ]

    # Select the most performant models in terms of recall
    # (within 1 sigma from the best)
    best_recall_std = high_precision_cv_results["mean_test_recall"].std()
    best_recall = high_precision_cv_results["mean_test_recall"].max()
    best_recall_threshold = best_recall - best_recall_std

    high_recall_cv_results = high_precision_cv_results[
        high_precision_cv_results["mean_test_recall"] > best_recall_threshold
    ]
    print(
        "Out of the previously selected high precision models, we keep all the\n"
        "the models within one standard deviation of the highest recall model:"
    )
    print_dataframe(high_recall_cv_results)

    # From the best candidates, select the fastest model to predict
    fastest_top_recall_high_precision_index = high_recall_cv_results[
        "mean_score_time"
    ].idxmin()

    print(
        "\nThe selected final model is the fastest to predict out of the previously\n"
        "selected subset of best models based on precision and recall.\n"
        "Its scoring time is:\n\n"
        f"{high_recall_cv_results.loc[fastest_top_recall_high_precision_index]}"
    )

    return fastest_top_recall_high_precision_index


def get_classifier_class_name(classifier):
    return classifier.__class__.__name__


def create_models_RF_grid(param_grid=None, testing=False):
    # hyperparams
    max_depths = [5, 7, 10, 15] #, 25, 50]
    num_estimators = [11, 15, 21, 51] #, 75, 100, 200] 
    criterions = ['gini', 'entropy'] 
    class_weights = [None, 'balanced', 'balanced_subsample']
    # class_weights = ['balanced']


    if testing:
        max_depths = [5]
        num_estimators = [9] 
        criterions = ['gini']
        class_weights = [None]


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "max_depth": max_depths,
            "n_estimators": num_estimators,
            "criterion": criterions,
            "class_weight": class_weights,
            "random_state": [RANDOM_STATE],
        }
    )    

    classifier = RandomForestClassifier()

    return classifier, param_grid


def create_models_DT_grid(param_grid=None, testing=False):
    # hyperparams
    max_depths = [3, 4, 5, 7, 9, 10, 15, 25] #, 50]
    criterions = ['gini', 'entropy'] #, 'log_loss'] LOG-LOSS DOESN'T WORK
    class_weights = [None, 'balanced']


    if testing:
        max_depths = [5]
        criterions = ['entropy']
        class_weights = [None]


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "max_depth": max_depths,
            "criterion": criterions,
            "class_weight": class_weights,
            "random_state": [RANDOM_STATE],
        }
    )    

    classifier = DecisionTreeClassifier()

    return classifier, param_grid


def create_models_NB_Complement_grid(param_grid=None, testing=False):
    # hyperparams
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    norms = [False, True]
    force_alphas = [False, True]

    if testing:
        alphas = [0.1]
        norms = [False]
        force_alphas = [False]


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "alpha": alphas,
            "norm": norms,
            "force_alpha": force_alphas,
        }
    )    

    classifier = ComplementNB()

    return classifier, param_grid


def create_models_kNN_grid(param_grid=None, testing=False):
    # hyperparams
    weights = ['uniform', 'distance']
    distance_metrics = [
        'euclidean',
        'manhattan',
        'chebyshev',
    ]
    #kNN
    ks = [3, 5, 9, 15] 


    if testing:
        weights = ['distance']
        distance_metrics = ['manhattan']
        #kNN
        ks = [5] 


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "n_neighbors": ks,
            "weights": weights,
            "metric": distance_metrics,
        }
    )    

    classifier = KNeighborsClassifier()

    return classifier, param_grid


def create_models_RadiusNN_grid(param_grid=None, testing=False):
    # hyperparams
    weights = ['uniform', 'distance']
    distance_metrics = [
        'euclidean',
        'manhattan',
        'chebyshev',
    ]

    #radius
    radius_set = [0.5, 0.7, 1.0, 1.5] 
    leaf_sizes = [50, 100, 200, 300, 500] 
    outlier_labels = [0, 1]

    if testing:
        weights = ['distance']
        distance_metrics = ['manhattan']
        #radius
        radius_set = [0.3] 
        leaf_sizes = [50] 
        outlier_labels = [1]


    if param_grid is None:
        param_grid = []


    param_grid.append(
        {
            "radius": radius_set,
            "weights": weights,
            "leaf_size": leaf_sizes,
            "outlier_label": outlier_labels,
            "metric": distance_metrics,
        }
    )    

    classifier = RadiusNeighborsClassifier()


    return classifier, param_grid


def create_models_NB_Gaussian_grid(param_grid=None, testing=False):

    param_grid = [{}]    

    classifier = GaussianNB()

    return classifier, param_grid


def create_models_NN_grid(qty_features, param_grid=None, testing=False):
    # hyperparams
    max_iter = [2000]
    layers = [
        (30),
        (30, 30),
        (30, 30, 30),
        (qty_features,),
        (qty_features, qty_features),
        (qty_features, qty_features, qty_features),
        (qty_features, (qty_features*2)),
        (qty_features, (qty_features*2), qty_features),
        (qty_features, (qty_features*2), (qty_features*2), qty_features),
    ]
    alphas = [0.0001, 0.00001, 0.05, 0.1, 0.3, 0.5]
    activations = ['tanh', 'relu']
    solvers = ['sgd', 'adam']
    learning_rates = ['constant','adaptive']
    learning_rate_init = [0.1, 0.01, 0.3, 0.03, 0.5, 0.7]


    if testing:
        max_iter = [500]
        layers = [(qty_features)]
        alphas = [0.1]
        activations = ['logistic']
        solvers = ['adam']
        learning_rates = ['constant']
        learning_rate_init = [0.1]


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "max_iter": max_iter,
            "hidden_layer_sizes": layers,
            "alpha": alphas,
            "activation": activations,
            "solver": solvers,
            "learning_rate": learning_rates,
            "learning_rate_init": learning_rate_init,
            "random_state": [RANDOM_STATE],
        }
    )    
   
    classifier = MLPClassifier()

    return classifier, param_grid


def create_models_BalancedBagging_grid(estimator, param_grid=None, testing=False):
    # hyperparams
    num_estimators = [7, 11, 15, 19, 21, 25, 31 ,51, 101]
    
    sampling_strategies = ['all', 'majority', 'auto']
    warm_starts = [False, True]
    replacements = [False, True] 


    if testing:
        num_estimators = [7] 
        warm_starts = [False]
        replacements = [False] 

    if param_grid is None:
        param_grid = []

    if type(estimator) is not list:
        estimator = [estimator]


    param_grid.append(
        {
            "estimator": estimator,
            "n_estimators": num_estimators,
            "sampling_strategy": sampling_strategies,
            "warm_start": warm_starts,
            "replacement": replacements,
            "random_state": [RANDOM_STATE],
        }
    )    

    classifier = BalancedBaggingClassifier()

    return classifier, estimator, param_grid


def create_models_BalancedRandomForest_grid(param_grid=None, testing=False):
    
    # parameters RF
    # hyperparams
    max_depths = [5, 7, 10, 15] #, 25, 50]
    criterions = ['gini', 'entropy'] 


    # hyperparams Balanced RF
    num_estimators = [7, 11, 15, 19, 21, 25, 31 ,51]
    
    sampling_strategies = ['all', 'majority', 'auto']
    warm_starts = [False, True]
    replacements = [False, True] 


    if testing:
        num_estimators = [7] 
        max_depths = [7]
        criterions = ['gini']
        warm_starts = [False]
        replacements = [False] 

    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            # RF
            "n_estimators": num_estimators,
            "max_depth": max_depths,
            "criterion": criterions,
            # Balanced RF
            "sampling_strategy": sampling_strategies,
            "warm_start": warm_starts,
            "replacement": replacements,
            "random_state": [RANDOM_STATE],
        }
    )    

    classifier = BalancedRandomForestClassifier()

    return classifier, param_grid



def create_models_SVM_grid(param_grid=None, testing=False):
    # hyperparams
    kernels = ['rbf', 'linear'] 
    gammas = ['scale', 'auto',]
    class_weights = [None, 'balanced',]
    Cs = [0.1, 0.5, 0.7, 1, 5, 7]

    if testing:
        kernels = ['linear']
        gammas = ['auto']
        class_weights = [None]
        Cs = [0.1]


    if param_grid is None:
        param_grid = []

    param_grid.append(
        {
            "C": Cs,
            "kernel": kernels,
            "gamma": gammas,
            "class_weight": class_weights,
            "probability": [True],
            "random_state": [RANDOM_STATE],
        }
    )    

    classifier = svm.SVC()

    return classifier, param_grid


# PERFORMANCE_THRESHOLD = 0.75 #(80%)
# PERFORMANCE_COLUMN    ='balanced_accuracy'

def get_grid_search_performances(grid, classifier, get_n_best_performances=-1):

    df_results = pd.DataFrame(grid.cv_results_)

    df_results['classifier'] = get_classifier_class_name(classifier)

    # reduce the name of the columns by removing the initial string "mean_test_"
    for col in df_results.columns:
        if col.startswith('mean_test_'):
            col_new = col.replace('mean_test_', '')
            df_results.rename(
                columns={col: col_new}, 
                inplace=True,
            )

    df_results.rename(columns={'mean_score_time': 'fit_time'}, inplace=True)

    # get only the columns of interest        
    cols_of_interest = [
        'classifier',
        'balanced_accuracy',
        'sensitivity',
        'specificity',
        'auc',
        'accuracy',
        'precision',
        'f1',
        'params',
        'fit_time'
    ]
    df_results = df_results[cols_of_interest]

    # rank the results by 'balanced_accuracy', 'sensitivity', 'specificity'
    df_results = sort_performances_results(
        df=df_results,
    )

    # get only the "n" best performances (default=5)
    if get_n_best_performances != -1:
        df_results = df_results.head(get_n_best_performances)

    # round the values using 2 decimal places
    df_results = df_results.round(2)        
    
    return df_results


def grid_search_refit_strategy(cv_results):
    """Define the strategy to select the best estimator.

    The strategy defined here is to filter-out all results below a precision threshold
    of 0.98, rank the remaining by recall and keep all models with one standard
    deviation of the best by recall. Once these models are selected, we can select the
    fastest model to predict.

    Parameters
    ----------
    cv_results : dict of numpy (masked) ndarrays
        CV results as returned by the `GridSearchCV`.

    Returns
    -------
    best_index : int
        The index of the best estimator as it appears in `cv_results`.
    """
    # balanced_accuracy_threshold = 0.73

    df_cv_results = pd.DataFrame(cv_results)
    # # print("All grid-search results:")
    # # print_dataframe(df_cv_results)

    # # Filter-out all results below the threshold
    # high_bal_acc_cv_results = df_cv_results[
    #     df_cv_results[f'mean_test_balanced_accuracy'] >= balanced_accuracy_threshold
    # ]

    # print(f"Models with a precision higher than {balanced_accuracy_threshold}:")
    # print_dataframe(high_precision_cv_results)
    # high_bal_acc_cv_results = df_cv_results

    df_cv_results = df_cv_results[
        [
            "mean_score_time",
            #
            "mean_test_balanced_accuracy",
            "std_test_balanced_accuracy",
            "rank_test_balanced_accuracy",
            #
            "mean_test_sensitivity",
            "std_test_sensitivity",
            "rank_test_sensitivity",
            #
            "mean_test_specificity",
            "std_test_specificity",
            "rank_test_specificity",
            #
            "params",
        ]
    ]

    # Select the most performant models in terms of balanced accuracy
    # (within 1 sigma from the best)
    best_bal_acc_std = df_cv_results["mean_test_balanced_accuracy"].std()
    best_bal_acc = df_cv_results["mean_test_balanced_accuracy"].max()
    best_bal_acc_threshold = best_bal_acc - best_bal_acc_std

    high_bal_acc_cv_results = df_cv_results[
        df_cv_results["mean_test_balanced_accuracy"] > best_bal_acc_threshold
    ]

    # Select the most performant models in terms of sensitivity
    # (within 1 sigma from the best)
    best_sensitivity_std = high_bal_acc_cv_results["mean_test_sensitivity"].std()
    best_sensitivity = high_bal_acc_cv_results["mean_test_sensitivity"].max()
    best_sensitivity_threshold = best_sensitivity - best_sensitivity_std

    high_sensitivity_cv_results = high_bal_acc_cv_results[
        high_bal_acc_cv_results["mean_test_sensitivity"] > best_sensitivity_threshold
    ]
    # print(
    #     "Out of the previously selected high precision models, we keep all the\n"
    #     "the models within one standard deviation of the highest recall model:"
    # )
    # print('ASSSSS')
    # print(high_sensitivity_cv_results)
    # print()

    # From the best candidates, select the fastest model to predict
    fastest_top_sensitivity_high_precision_index = high_sensitivity_cv_results[
        "mean_score_time"
    ].idxmin()

    # print(
    #     "\nThe selected final model is the fastest to predict out of the previously\n"
    #     "selected subset of best models based on precision and recall.\n"
    #     "Its scoring time is:\n\n"
    #     f"{high_recall_cv_results.loc[fastest_top_sensitivity_high_precision_index]}"
    # )

    print(high_sensitivity_cv_results)

    return fastest_top_sensitivity_high_precision_index


def create_classifier_from_string(classifier_as_str, dict_params_as_str):
    # convert params to dict
    try:
        dict_params = dict(dict_params_as_str)
    except:
        dict_params = eval(dict_params_as_str)    

    # create an model instance passing the hyperparameters
    klass = globals()[classifier_as_str]
    clf = klass(**dict_params)

    return clf



def get_performances_from_predictions(y_validation, y_pred, y_pred_proba):
    # calculate the scores using y_pred
    bal_acc = np.round(balanced_accuracy_score(y_validation, y_pred), 2)
    sens    = np.round(recall_score(y_validation, y_pred), 2)
    spec    = np.round(recall_score(y_validation, y_pred, pos_label=0), 2)
    f1      = np.round(f1_score(y_validation, y_pred), 2)
    acc     = np.round(accuracy_score(y_validation, y_pred), 2)
    precision    = np.round(precision_score(y_validation, y_pred), 2)

    # calculate AUC using y_pred_proba
    auc     = np.round(roc_auc_score(y_validation, y_pred_proba), 2)

    return bal_acc, sens, spec, auc, acc, precision, f1



DEFAULT_SCORE = 'balanced_accuracy'

def exec_grid_search(classifier, param_grid, X_train, y_train, 
                     X_valid, y_valid, 
                     cv=None, 
                     n_jobs=N_JOBS, verbose=1, scoring=None, 
                     refit=None, return_train_score=False,
                     get_n_best_performances=5,
                    #  sort_results=True, 
                    #  dataset_info='', 
                    #  features_info='', 
                    #  plot_roc_curve=False,
                     ):


    # get only array of output y_train, if it was a dataFrame
    if type(y_train) is pd.DataFrame:
        y_train = y_train[utils.CLASS_COLUMN].ravel()  

    # # get only array of output y_valid, if it was a dataFrame
    # if type(y_valid) is pd.DataFrame:
    #     y_valid = y_valid[utils.CLASS_COLUMN].ravel()  

    if scoring is None:
        scoring = get_default_scoring()

    # define the default REFIT, if not informed
    if refit is None:
        refit = DEFAULT_SCORE

    # define the default kFold object, if not informed
    if cv is None:
        cv = get_kfold_splits()


    # create object GridSearch
    grid = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid, 
        scoring=scoring,
        cv=cv, 
        verbose=verbose,
        n_jobs=n_jobs,
        return_train_score=return_train_score,
        refit=refit,
    )

    # train the gridSearch models
    grid.fit(X_train, y_train)    


    # get performance for each set of hyperparams
    # filtering by the "n" best performances (default: 5)
    df_performances = get_grid_search_performances(
        classifier=classifier,
        grid=grid,
        get_n_best_performances=get_n_best_performances,
    )


    best_models_performances = []
    det_curve_data = []
    precision_recall_curve_data = []
    roc_curve_data = []
    predictions_data = []

    # make predictions using the best classifiers
    for idx, row in df_performances.iterrows():
        clf = create_classifier_from_string(
            classifier_as_str=row.classifier,
            dict_params_as_str=row.params,
        )

        # fit the classifier again using training data
        clf.fit(X_train, y_train)

        # make predictions
        y_pred = clf.predict(X_valid)

        # make predictions using probabilities, returning the class 
        # probabilities for sample.
        # The first column represents the probability of the 
        # negative class (Non-Short) and the second column represents 
        # the probability of the positive class (Short).
        y_pred_proba = clf.predict_proba(X_valid)
        y_pred_proba = y_pred_proba[:,1] # get short-survival probabilities

        bal_acc, sens, spec, auc, acc, prec, f1 = get_performances_from_predictions(
            y_validation=y_valid,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
        )


        model_desc = utils.get_model_description(row.classifier)
        params = str(row.params).replace('\n', '')

        # remove estimator param from the Balanced-Bagging classifier
        if model_desc == 'Balanced Bagging':
            estimator = str(row.params.pop('estimator')).replace('\n', '')
            params = str(row.params).replace('\n', '')

            best_models_performances.append({
                'Model': model_desc,
                'balanced_accuracy': bal_acc,
                'sensitivity': sens,
                'specificity': spec,
                'f1_score': f1,
                'AUC': auc,
                'accuracy': acc,
                'precision': prec,
                'Model_Class': row.classifier,
                'Hyperparams': params,
                'Estimator': estimator,
                'fit_time': row.fit_time,
            })

        else:
            best_models_performances.append({
                'Model': model_desc,
                'balanced_accuracy': bal_acc,
                'sensitivity': sens,
                'specificity': spec,
                'f1_score': f1,
                'AUC': auc,
                'accuracy': acc,
                'precision': prec,
                'Model_Class': row.classifier,
                'Hyperparams': params,
                'fit_time': row.fit_time,
            })

    # create a dataFrame with the best performances
    df_best_performances = pd.DataFrame(best_models_performances)

    # sort the performances and get the first row (best model)
    df_best_performances = sort_performances_results(
        df=df_best_performances,
    )

    df_best_performances_detailed = df_best_performances.copy()

    df_best_performances = df_best_performances.head(1)

    # collect additional info about the best model
    for idx, row in df_best_performances.iterrows():

        # =======================================================
        # Detection Error Tradeoff (DET) curve
        # =======================================================
        fpr, fnr, thresholds = sk.metrics.det_curve(y_valid, y_pred_proba)
        det_curve_aux = {
            'Model'      : row.Model, 
            'Hyperparams': row.Hyperparams, 
            'FPR'        : fpr, 
            'FNR'        : fnr, 
            'Thresholds' : thresholds,
        }
        det_curve_data.append(det_curve_aux)

        # =======================================================
        # Precision-Recall curve
        # =======================================================
        precision, recall, thresholds = sk.metrics.precision_recall_curve(y_valid, y_pred_proba)
        au_prec_recall_curve = sk.metrics.auc(recall, precision)
        precision_recall_curve_aux = {
            'Model'      : row.Model, 
            'Hyperparams': row.Hyperparams, 
            'Precision'  : precision,
            'Recall'     : recall, 
            'Thresholds' : thresholds,
        }
        precision_recall_curve_data.append(precision_recall_curve_aux)

        # =======================================================
        # ROC curve
        # =======================================================
        fpr, tpr, thresholds = sk.metrics.roc_curve(y_valid, y_pred_proba)
        roc_auc = sk.metrics.auc(fpr, tpr)
        roc_curve_aux = {
            'Model'      : row.Model, 
            'Hyperparams': row.Hyperparams, 
            'FPR'        : fpr, 
            'TPR'        : tpr, 
            'Thresholds' : thresholds,
        }
        roc_curve_data.append(roc_curve_aux)

        # =======================================================
        # Predictions data (using predict and predict_proba)
        # =======================================================
        predictions_aux = {
            'Model'       : row.Model, 
            'Hyperparams' : row.Hyperparams, 
            'y_pred'      : y_pred, 
            'y_pred_proba': y_pred_proba
        }
        predictions_data.append(predictions_data)



    # create a dict representing additional info (DET, Prec-Recall-curve, ROC, and predictions)
    additional_info = {
        'DET': det_curve_data,
        'Precision-Recall-Curve': precision_recall_curve_data,
        'ROC': roc_curve_data,
        'Predictions': predictions_data,
    }



    return grid, df_best_performances, df_best_performances_detailed, additional_info




def exec_grid_search_and_save_performances(dir_dest, testing, grid, classifier, scenario, features_config, 
                X_train, y_train, X_valid, y_valid):


    # Create folders if not exists
    if not os.path.exists(dir_dest):
        os.makedirs(dir_dest)

    dir_serialized_data = f'{dir_dest}/serialized_data'    
    if not os.path.exists(dir_serialized_data):
        os.makedirs(dir_serialized_data)


    # get model description
    model_desc = utils.get_model_description(classifier).replace('-', '')

    # fit using CV with the trainning data
    grid.fit(X_train, y_train.values.ravel())

    # get performance for each set of hyperparams
    df_trainning_performances = get_grid_search_performances(
        classifier=classifier,
        grid=grid,
        get_n_best_performances=-1, # get all performances
    )
    
    df_trainning_performances.insert(0, 'Scenario', scenario)
    df_trainning_performances.insert(1, 'Features', features_config)
    df_trainning_performances.insert(2, 'Model', model_desc)

    df_trainning_performances.insert(
        len(df_trainning_performances.columns)-1, 
        'Model_Class', 
        df_trainning_performances['classifier']
    )

    df_trainning_performances.insert(
        len(df_trainning_performances.columns)-1, 
        'Hyperparams', 
        df_trainning_performances['params']
    )



    # ==============================================
    # save predictions results using test data
    # ==============================================
    
    additional_info = []
    validation_performances = []

    # make predictions using the best classifiers
    for idx, row in df_trainning_performances.iterrows():
        clf = create_classifier_from_string(
            classifier_as_str=row.classifier,
            dict_params_as_str=row.params,
        )

        # fit the classifier again using training data
        clf.fit(X_train, y_train)

        # make predictions
        y_pred = clf.predict(X_valid)

        # make predictions using probabilities, returning the class 
        # probabilities for sample.
        # The first column represents the probability of the 
        # negative class (Non-Short) and the second column represents 
        # the probability of the positive class (Short).
        y_pred_proba = clf.predict_proba(X_valid)
        y_pred_proba = y_pred_proba[:,1] # get short-survival probabilities

        bal_acc, sens, spec, auc, acc, prec, f1 = get_performances_from_predictions(
            y_validation=y_valid,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
        )

        classif_desc = utils.get_model_description(row.classifier)
        params = str(row.params).replace('\n', '').replace(' ', '')
        estimator = '' # used only with BalancedBagging
        estimator_params = ''
        
        # remove the estimator param info for the Balanced-Bagging classifier
        if str(row.classifier) == 'BalancedBaggingClassifier':
            estimator = row.params.pop('estimator')
            estimator_params = str(estimator.get_params()).replace('\n', '').replace(' ', '')
            # extract name of estimator classifier
            estimator = str(estimator).replace('\n', '').replace(' ', '').split('(')[0]
        #
        elif str(row.classifier) == 'BalancedRandomForestClassifier':
            estimator = 'RandomForest'


        validation_performances.append({
            'Scenario': scenario,
            'Features': features_config,
            'Model': classif_desc,
            # validation performance
            'balanced_accuracy': bal_acc,
            'sensitivity': sens,
            'specificity': spec,
            'f1_score': f1,
            'AUC': auc,
            'accuracy': acc,
            'precision': prec,
            #
            'Model_Class': row.classifier,
            'Hyperparams': params,
            'Estimator': utils.get_model_description(estimator),
            'Estimator_Class': estimator,
            'Estimator_Hyperparams': estimator_params,
            'fit_time': row.fit_time,
            # trainning performance
            'train_balanced_accuracy': row.balanced_accuracy,
            'train_sensitivity': row.sensitivity,
            'train_specificity': row.specificity,
            'train_f1_score': row.f1,
            'train_AUC': row.auc,
            'train_accuracy': row.accuracy,
            'train_precision': row.precision,
        })


        # get and save information about DET curve, ROC curve, and Precision-Recal curve
        det_curve, roc_curve, prec_recall_curve = get_predictions_additional_info(
            y_valid, 
            y_pred_proba
        )
        #
        additional_info.append({
            'Scenario': scenario,
            'Features': features_config,
            'Model': classif_desc,
            'Model_Class': row.classifier,
            'Hyperparams': params,
            'Estimator': utils.get_model_description(estimator),
            'Estimator_Class': estimator,
            'Estimator_Hyperparams': estimator_params,
            'DET': det_curve,
            'ROC_Curve': roc_curve,
            'Prec_Recall_Curve': prec_recall_curve,
            'y_pred': y_pred, 
            'y_pred_proba': y_pred_proba,
        })



    # create a dataFrame with the best performances
    df_validation_performances = pd.DataFrame(validation_performances)


    # ==================================================
    # define the name used to save CSV and Pickle files
    # ==================================================
    file_prefix = 'TESTING__' if testing else ''
    if str(classifier) in ['BalancedBaggingClassifier()', 'BalancedRandomForestClassifier()']:
        model_abrev_desc = str(estimator).replace('Classifier', '').replace('()', '').replace('-', '').replace('.', '').replace(' ', '')
        suffix = utils.get_model_short_description(classifier).replace('-', '').replace('.', '').replace(' ', '')
        name_to_save = f'{file_prefix}{model_abrev_desc}__{features_config}__{scenario}__{suffix}'
    else:
        model_abrev_desc = str(classifier).replace('Classifier', '').replace('()', '').replace('-', '').replace('.', '').replace(' ', '')
        name_to_save = f'{file_prefix}{model_abrev_desc}__{features_config}__{scenario}'



    # sort the validation performances
    df_validation_performances = sort_performances_results(
        df=df_validation_performances,
    )


    # =========================================
    # save CV validation + trainning results
    # ==========================================
    print('SAVING PERFORMANCE RESULTS...')
    csv_to_save = f'{dir_dest}/performance__{name_to_save}.csv'
    utils.save_to_csv(df=df_validation_performances, csv_file=csv_to_save)



    if not testing:
        # =========================================
        # save (serialize) the gridSearch instance
        # ==========================================
        grid_object_to_save = f'{dir_serialized_data}/grid_search__{name_to_save}.pickle'

        print('SAVING GRID-SEARCH OBJECT...')
        with open(grid_object_to_save, 'wb') as handle:
            pickle.dump(grid, handle) #, protocol=pickle.HIGHEST_PROTOCOL)


        # =========================================
        # save (serialize) the additional_info data (DET, ROC, and Prec-Recal curves)
        # ==========================================
        add_info_to_save = f'{dir_serialized_data}/additional_info__{name_to_save}.pickle'
        with open(add_info_to_save, 'wb') as handle:
            pickle.dump(additional_info, handle) #, protocol=pickle.HIGHEST_PROTOCOL)


    #
    return grid, df_validation_performances


def get_predictions_additional_info(y_valid, y_pred_proba):

    # =======================================================
    # Detection Error Tradeoff (DET) curve
    # =======================================================
    fpr, fnr, thresholds = sk.metrics.det_curve(y_valid, y_pred_proba)
    det_curve = {
        'FPR'        : fpr, 
        'FNR'        : fnr, 
        'Thresholds' : thresholds,
    }

    # =======================================================
    # ROC curve
    # =======================================================
    fpr, tpr, thresholds = sk.metrics.roc_curve(y_valid, y_pred_proba)
    roc_curve = {
        'FPR'        : fpr, 
        'TPR'        : tpr, 
        'Thresholds' : thresholds,
    }

    # =======================================================
    # Precision-Recall curve
    # =======================================================
    precision, recall, thresholds = sk.metrics.precision_recall_curve(y_valid, y_pred_proba)
    prec_recall_curve = {
        'Precision'  : precision,
        'Recall'     : recall, 
        'Thresholds' : thresholds,
    }

    #
    return det_curve, roc_curve, prec_recall_curve



def create_model_instances_from_performances(df):
    models = list()
    for idx, row in df.iterrows():
        model_instance = create_classifier_from_string(
            classifier_as_str=row.Model_Class,
            dict_params_as_str=row.Hyperparams,
        )
        models.append(model_instance)

    return models






def corrected_std(differences, n_train, n_test):
    """Corrects standard deviation using Nadeau and Bengio's approach.

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    """
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std


def compute_corrected_ttest(differences, df, n_train, n_test):
    """Computes right-tailed paired t-test with corrected variance.

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    """
    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)

    if std ==0:
        std = 0.001
    # print(mean, std)

    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val



def get_best_model_instances():
    best_classifiers = list()

    # Decision Tree,Ensemble-Imbalance,All Features,Balanced Bagging,0.88,0.96,0.81,0.57,0.94,0.82,0.4,
    estimator = DecisionTreeClassifier(class_weight='balanced',max_depth=4,random_state=42)
    dt_ei = BalancedBaggingClassifier(
        estimator=estimator,
    #     ':DecisionTreeClassifier(class_weight='balanced',max_depth=4,random_state=42),
        **{'n_estimators':7,'random_state':42,'replacement':True,'sampling_strategy':'all','warm_start':True}
    )
    best_classifiers.append(['Decision Tree (EI)', 'Ensemble-Imbalance', dt_ei])

    # Decision Tree,Single-Model,All Features,Decision Tree,0.83,0.85,0.81,0.53,0.88,0.82,0.38,
    dt_sm = DecisionTreeClassifier(
        **{'class_weight':'balanced','criterion':'gini','max_depth':4,'random_state':42}
    )
    best_classifiers.append(['Decision Tree (SM)', 'Single-Model', dt_sm])



    # Neural Networks,Ensemble-Imbalance,All Features,Balanced Bagging,0.88,0.96,0.8,0.56,0.93,0.82,0.39,
    estimator = MLPClassifier(activation='tanh',alpha=0.1,hidden_layer_sizes=30,learning_rate='adaptive',learning_rate_init=0.7,max_iter=2000,random_state=42)
    nn_ei = BalancedBaggingClassifier(
        estimator=estimator,
        **{'n_estimators':101,'random_state':42,'replacement':True,'sampling_strategy':'auto','warm_start':False}
    )
    best_classifiers.append(['Neural Networks (EI)', 'Ensemble-Imbalance', nn_ei])

    # Neural Networks,Single-Model,All Features,Neural Networks,0.87,0.87,0.87,0.61,0.94,0.87,0.47,
    nn_sm = MLPClassifier(
        **{'activation':'tanh','alpha':0.3,'hidden_layer_sizes':(23,23,23),'learning_rate':'constant',
        'learning_rate_init':0.7,'max_iter':2000,'random_state':42,'solver':'sgd'}
    )
    best_classifiers.append(['Neural Networks (SM)', 'Single-Model', nn_sm])



    # Random Forest,Ensemble-Imbalance,All Features,Balanced Random Forest,0.87,0.96,0.79,0.55,0.93,0.81,0.38,
    rf_ei = BalancedRandomForestClassifier(
        **{'criterion':'entropy','max_depth':7,'n_estimators':19,'random_state':42,'replacement':True,
        'sampling_strategy':'auto','warm_start':False}
    )
    best_classifiers.append(['Random Forest (EI)', 'Ensemble-Imbalance', rf_ei])

    # Random Forest,Single-Model,All Features,Random Forest,0.87,0.89,0.86,0.6,0.92,0.86,0.46,
    rf_sm = RandomForestClassifier(
        **{'class_weight':'balanced','criterion':'gini','max_depth':5,'n_estimators':51,'random_state':42}
    )
    best_classifiers.append(['Random Forest (SM)', 'Single-Model', rf_sm])


    # SVM,Ensemble-Imbalance,All Features,Balanced Bagging,0.87,0.94,0.8,0.54,0.92,0.81,0.38,
    estimator = SVC(C=0.7,class_weight='balanced',gamma='auto',probability=True,random_state=42)
    svm_ei = BalancedBaggingClassifier(
        estimator=estimator,
        **{'n_estimators':51,'random_state':42,'replacement':False,'sampling_strategy':'auto','warm_start':True}
    )
    best_classifiers.append(['SVM (EI)', 'Ensemble-Imbalance', svm_ei])

    # SVM,Single-Model,All Features,SVM,0.86,0.91,0.81,0.55,0.93,0.82,0.39,
    {'C':0.7,'class_weight':'balanced','gamma':'auto','kernel':'rbf','probability':True,'random_state':42}
    svm_sm = SVC(
        **{'C':0.7,'class_weight':'balanced','gamma':'auto','kernel':'rbf','probability':True,'random_state':42}
    )
    best_classifiers.append(['SVM (SM)', 'Single-Model', svm_sm])




    # Naïve Bayes,Ensemble-Imbalance,All Features,Balanced Bagging,0.84,0.85,0.83,0.54,0.9,0.83,0.4
    estimator = GaussianNB()
    nb_ei = BalancedBaggingClassifier(
        estimator = estimator,
        **{'n_estimators':31,'random_state':42,'replacement':True,'sampling_strategy':'all','warm_start':False}
    )
    best_classifiers.append(['Naïve Bayes (EI)', 'Ensemble-Imbalance', nb_ei])

    # Naïve Bayes,Single-Model,All Features,Naïve Bayes,0.8,0.74,0.86,0.53,0.9,0.84,0.41,
    nb_sm = GaussianNB()
    best_classifiers.append(['Naïve Bayes (SM)', 'Single-Model', nb_sm])


    # k-NN,Ensemble-Imbalance,All Features,Balanced Bagging,0.85,0.85,0.85,0.57,0.9,0.85,0.43,
    estimator = KNeighborsClassifier(metric='euclidean',weights='distance')
    knn_ei = BalancedBaggingClassifier(
        estimator=estimator,
        **{'n_estimators':101,'random_state':42,'replacement':True,'sampling_strategy':'all',
        'warm_start':True}
    )
    best_classifiers.append(['k-NN (EI)', 'Ensemble-Imbalance', knn_ei])

    # k-NN,Single-Model,All Features,k-NN,0.68,0.4,0.96,0.48,0.81,0.9,0.59,
    knn_sm = KNeighborsClassifier(
        **{'metric':'manhattan','n_neighbors':3,'weights':'uniform'}
    )
    best_classifiers.append(['k-NN (SM)', 'Single-Model', knn_sm])


    return best_classifiers



