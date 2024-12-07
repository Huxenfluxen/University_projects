# treebased_methods

import lightgbm as lgb
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from skopt import BayesSearchCV
# from skopt.callbacks import VerboseCallback
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler

'''Light Gradient Boost Method. Taking data already splitted into training and test parts'''
def lgb_method(X, y, n_splits=3, repeatedCV=True, bayes=False, randomize=200, verbosity=3, test_size=0.2):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params_grid = {
        'max_depth': [3, 5, 7, 10, 12, 15, 20, 25, 30],
        'n_estimators': [50, 100, 200, 500, 1000],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1],
        # 'num_leaves': [15, 31, 50, 70],
        'learning_rate': [5e-4, 0.001, 0.01, 0.05, 0.1, 0.5],
        # 'feature_fraction': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
        # 'bagging_fraction': [0.6, 0.75, 0.9, 1],
        # 'bagging_freq': [0, 5, 10, 15],
        'lambda_l1': [0, 0.1, 0.5, 1, 10],  # L1 regularization
        'lambda_l2': [0, 0.1, 0.5, 1, 10],   # L2 regularization
        # 'min_data_in_leaf': [20, 40, 60, 80, 100],
        # 'min_split_gain': [0, 0.1, 0.5],
        'min_child_weight': [1, 5, 10]
        }
    bayes_grid = {
        'max_depth': [5, 7, 10, 12, 15],
        'n_estimators': [50, 100, 150, 200, 300],
        'subsample': (0.6, 0.9),
        'num_leaves': [15, 31, 50],
        'learning_rate': (1e-3, 0.1, 'log-uniform'),
        'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
        'bagging_fraction': [0.6, 0.8, 1.0],
        'bagging_freq': [0, 5, 10],
        'lambda_l1': (0, 10),  # L1 regularization
        'lambda_l2': (0, 10),   # L2 regularization
        'min_data_in_leaf': [20, 40, 60, 80, 100],
        'min_split_gain': (0, 0.5),
        'min_child_weight': (1, 5)
        }
    
    lgb_est = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', boosting_type='gbdt', verbose=-1)
    
    if repeatedCV:
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) # Cross validation
    
    if bayes:
        cv_lgbm = BayesSearchCV(estimator=lgb_est, search_spaces=bayes_grid, n_iter=100, cv=skf, n_jobs=-1, verbose=10, random_state=42)
    elif randomize > 0:
        cv_lgbm = RandomizedSearchCV(lgb_est, param_distributions=params_grid, n_iter=randomize, cv=skf, verbose=verbosity, random_state=42, n_jobs=-1)
    else:
        cv_lgbm = GridSearchCV(lgb_est, params_grid, cv=skf, n_jobs=-1, verbose=verbosity)
        
    cv_lgbm.fit(X_train, y_train)
    
    lgbm_params = cv_lgbm.best_params_
    lgbm_accuracy = cv_lgbm.best_score_
    lgbm_results = cv_lgbm.cv_results_
    
    print(f"LightGBM accuracy on training data: {lgbm_accuracy}\n"
          f"Best LightGBM parameters: {lgbm_params}")
    
    y_pred = cv_lgbm.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    final_model = cv_lgbm.best_estimator_
    final_model.fit(X, y)
    
    print(f"LightGBM accuracy on test data: {test_accuracy}")
    
    return final_model, lgbm_results
    

'''Random forest classifier. Taking data already splitted into training and test parts
    two extra parameters for using bagging and/or randomized grid search'''
def rfc_method(X, y, n_splits=3, bagging=True, randomize=200, verbosity=3, test_size=0.2):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    rfc = RandomForestClassifier()
    classifier = AdaBoostClassifier(rfc, random_state=42)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if bagging:
        classifier = BaggingClassifier(classifier, n_estimators=10, random_state=42)
        param_grid = {"classifier__estimator__estimator__n_estimators": [50, 100, 150, 200],
                    "classifier__estimator__estimator__max_depth": [5, 10, 15, 20],
                    "classifier__estimator__n_estimators": [50, 100, 200],
                    "classifier__estimator__learning_rate": [0.001, 0.1, 0.5],
                    'pca__n_components': [0.9, 0.93, 0.95, 0.98]
                    }
    else:
        param_grid = {"classifier__estimator__n_estimators": [50, 100, 150, 200],
                    "classifier__estimator__max_depth": [5, 10, 15, 20],
                    "classifier__n_estimators": [50, 100, 200],
                    "classifier__learning_rate": [0.001, 0.1, 0.5],
                    'pca__n_components': [0.9, 0.93, 0.95, 0.98]
                    }
    pipe = Pipeline([       # Defining the classifier using principal component analysis to avoid multicollinearity as much as possible
            ('scaler', StandardScaler()),   # Scaling features
            ('pca', PCA(random_state=42)),
            ('classifier', classifier)
            ])
    if randomize > 0:
        cv_forest = RandomizedSearchCV(pipe, param_distributions=param_grid, n_iter=randomize, cv=skf, n_jobs=-1, verbose=verbosity, random_state=42)
    else:
        cv_forest = GridSearchCV(pipe, param_grid, cv=skf, n_jobs=-1, verbose=verbosity)
        
    cv_forest.fit(X_train, y_train)

    rfc_params = cv_forest.best_params_
    train_accuracy = cv_forest.best_score_
    rfc_results = cv_forest.cv_results_

    print(f"Forest accuracy on training data: {train_accuracy}\n"
          f"Best forest parameters: {rfc_params}")

    y_pred = cv_forest.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    final_model = cv_forest.best_estimator_
    final_model.fit(X, y)

    print(f"Forest accuracy on test data: {test_accuracy}")
    
    return final_model, rfc_results

'''X Gradient Boost Method. Taking data already splitted into training and test parts. Basically the same as lgb.'''
def xgb_method(X, y, n_splits=3, repeatedCV=True, randomize=200, verbosity=3, test_size=0.2):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    params_grid = {
        'max_depth': [3, 5, 7, 10, 12, 15, 20, 25, 30],
        'n_estimators': [50, 100, 200, 500, 1000],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1],
        'learning_rate': [5e-4, 0.001, 0.01, 0.05, 0.1, 0.5],
        'reg_alpha': [0, 0.1, 0.5, 1, 10],  # L1 regularization
        'reg_lambda': [0, 0.1, 0.5, 1, 10],   # L2 regularization
        'min_child_weight': [1, 5, 10]
        }
    
    xgb_est = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', verbosity=2)
    xgb.cv()
    # Cross validation
    if repeatedCV:
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if randomize > 0:
        cv_xgb = RandomizedSearchCV(xgb_est, param_distributions=params_grid, n_iter=randomize, cv=skf, verbose=verbosity, random_state=42, n_jobs=-1)
    else:
        cv_xgb = GridSearchCV(xgb_est, params_grid, cv=skf, n_jobs=-1, verbose=verbosity)
        
    cv_xgb.fit(X_train, y_train)
    
    xgb_params = cv_xgb.best_params_
    xgb_accuracy = cv_xgb.best_score_
    xgb_results = cv_xgb.cv_results_
    
    print(f"XGB accuracy on training data: {xgb_accuracy}\n"
          f"Best XGB parameters: {xgb_params}")
    
    y_pred = cv_xgb.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    final_model = cv_xgb.best_estimator_
    final_model.fit(X, y)
    
    print(f"XGB accuracy on test data: {test_accuracy}")
    
    return final_model, xgb_results
    