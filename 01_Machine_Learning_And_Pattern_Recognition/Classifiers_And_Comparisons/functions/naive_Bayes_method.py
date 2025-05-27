# naive_Bayes.py

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
import numpy as np

def nb_method(X, y, n_splits=5, repeatedCV=True, randomize=200, verbosity=2, test_size=0.2):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    pipe = Pipeline([       # Defining the classifier using principal component analysis to avoid multicollinearity as much as possible
            ('scaler', StandardScaler()),   # Scaling features
            ('select', SelectKBest(score_func=f_classif)),
            ('pca', PCA(random_state=42)),
            ('lasso', SelectFromModel(Lasso(random_state=42))),
            ('nb', GaussianNB())
            ])
    
    params_grid = {
            'select__k': [5, 7, 9, 10, 'all'],
            'pca__n_components': [5, 7, 9, 11, 'mle'],
            'lasso__estimator__alpha': [0.01, 0.1, 1.0, 10]
            }
    
    if repeatedCV:
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if randomize > 0:
        cv_NB = RandomizedSearchCV(pipe, params_grid, cv=skf, n_jobs=1, verbose=verbosity, n_iter=randomize, random_state=42)
    else:
        cv_NB = GridSearchCV(pipe, params_grid, cv=skf, n_jobs=1, verbose=verbosity)
        
    cv_NB.fit(X_train, y_train)
    
    nb_params = cv_NB.best_params_
    train_accuracy = cv_NB.best_score_
        
    print(f"Naive Bayes accuracy on training data: {train_accuracy}\n"
          f"Best Naive Bayes parameters: {nb_params}")
    
    # Combining the train and test data to train the final model using the best parameters found
    # X = np.vstack((X_train, X_test))
    # y = np.hstack((y_train, y_test))
    
    y_pred = cv_NB.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    final_model = cv_NB.best_estimator_
    final_model.fit(X, y)
    
    print(f"Naive Bayes accuracy on test data: {test_accuracy}")
    
    return final_model