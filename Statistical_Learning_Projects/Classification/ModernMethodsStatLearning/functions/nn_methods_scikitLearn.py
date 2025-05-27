# nn_methods_scikitLearn

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, train_test_split, RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
import numpy as np

def NN_method(X, y, n_splits=3, repeatedCV=True, randomize=200, verbosity=3, test_size=0.2):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    params_grid = {
    'feature_selection__k': [3, 5, 8, 10, 11],
    'model__hidden_layer_sizes': [(50,), (100,), (1000,), (50, 50), (100, 100), (200, 100, 150)],
    'model__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'model__solver': ['adam', 'sgd'],
    'model__activation': ['relu', 'tanh', 'logistic'],
    'model__max_iter': [200, 400, 600],
    'model__alpha': [0.0001, 0.001, 0.01, 0.1],
    }
    
    pipe = Pipeline([
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('model', MLPClassifier(random_state=42))
    ])
    
    if repeatedCV:
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    
    if randomize > 0:
        cv_nn = RandomizedSearchCV(pipe, param_distributions=params_grid, n_iter=randomize, cv=skf, verbose=verbosity, random_state=42, n_jobs=-1)
    else:
        cv_nn = GridSearchCV(pipe, params_grid, cv=skf, n_jobs=-1, verbose=verbosity)

    cv_nn.fit(X_train, y_train)
    
    nn_params = cv_nn.best_params_
    nn_accuracy = cv_nn.best_score_
    nn_results = cv_nn.cv_results_
    
    print(f"Neural network accuracy on training data: {nn_accuracy}\n"
        f"Best neural network parameters: {nn_params}")
    
    y_pred = cv_nn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    final_model = cv_nn.best_estimator_
    final_model.fit(X, y)
    
    print(f"Neural network accuracy on test data: {test_accuracy}")
    
    return final_model, nn_results