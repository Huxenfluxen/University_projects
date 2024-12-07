# Stacking_methods.py

import treebased_methods as tbm
import svm_methods as svm
import knn_methods as kNN
import NN_methods as NNm
import nn_methods_scikitLearn as NNmSci
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from sklearn.svm import SVC



def stacking_method(X, y, n_splits=3, repeatedCV=True):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Prefitting the models on training data via methods defined in respective learning algo file
    # knn_model = kNN.knn_method(X_train, y_train, n_splits=n_splits, repeatedCV=False, randomize=150, verbosity=0, test_size=1)
    # xgb_model = tbm.xgb_method(X_train, y_train, n_splits=n_splits, repeatedCV=False, randomize=150, verbosity=0, test_size=1)
    # svm_model = svm.svm_method(X_train, y_train, n_splits=n_splits, randomize=150, verbosity=0, test_size=1)
    # ### Cannot use the NN_methods since it does not inherit from scikit learn ###
    # # NN_model = NNm.NN_method(X_train, y_train, n_splits=n_splits, repeatedCV=False, randomize=100, verbosity=0)
    # NN_model = NNmSci.NN_method(X_train, y_train, n_splits=n_splits, randomize=100, verbosity=0, test_size=1)
    
    
    # base_models = [
    #     ('knn', knn_model),
    #     ('svm', svm_model),
    #     ('tree', xgb_model),
    #     ('nn', NN_model)
    # ]
    
    
    
    base_models2 = [
        ('knn', KNeighborsClassifier()),
        ('svm', SVC()),
        ('tree', lgb.LGBMClassifier()),
        ('nn', MLPClassifier())
    ]
    
    if repeatedCV:
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    meta_model = LogisticRegression()
    # stack_classifier = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv="prefit", verbose=3)
    stack_classifier2 = StackingClassifier(estimators=base_models2, final_estimator=LogisticRegression(), cv=skf, verbose=3)
    
    #stack_results = cross_validate(stack_classifier2, X_train, y_train, cv=skf, return_train_score=True, verbose=3)
    
    # stack_classifier.fit(X_train, y_train)
    stack_classifier2.fit(X_train, y_train)
    
    # y_pred = stack_classifier.predict(X_test)
    y_pred2 = stack_classifier2.predict(X_test)
    
    # accuracy = accuracy_score(y_test, y_pred)
    accuracy2 = accuracy_score(y_test, y_pred2)
    
    # print(f"Stack accuracy on test data: {accuracy}")
    print(f"Stack accuracy 2 on test data: {accuracy2}")
    
    # final_model = stack_classifier.final_estimator_
    final_model2 = stack_classifier2.final_estimator_
    
    # final_model.fit(X, y)
    final_model2.fit(X, y)
    
    return final_model2#, stack_results