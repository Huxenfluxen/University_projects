# knn_methods.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np

def knn_method(X, y, n_splits=3, repeatedCV=True, randomize=200, verbosity=2, test_size=0.2):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    pipe = Pipeline([       # Defining the classifier using principal component analysis to avoid multicollinearity as much as possible
            ('scaler', StandardScaler()),   # Scaling features
            ('select', SelectKBest(score_func=f_classif, k='all')),
            #('pca', PCA(random_state=42)),
            ('knn', KNeighborsClassifier())
            ])
    neighb = list(np.cumsum(np.ones(75)).astype(dtype=int))[9:]
    pca_comp = list(np.cumsum(np.ones(11)).astype(dtype=int)) + ['mle']
    params_grid = {              # Defining the kernels to use 
            'knn__n_neighbors': neighb[10:],#[16, 17, 20],        # number of neigbours
            'knn__weights': ['uniform', 'distance'],    # Smoothness of decision boundary
            #'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'knn__metric': ['manhattan', 'euclidean', 'chebyshev'],#['minkowski'],
            # 'knn__metric_params': [{'p': 1}, {'p': 2}, {'p': np.inf}],
            #'pca__n_components': pca_comp[8:] #[7, 8, 9, 10, 11, 'mle'],
            'select__k': [6, 8, 9, 10, 11]
            }
    
    if repeatedCV:
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=42)
    else:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    if randomize > 0:
        cv_knn = RandomizedSearchCV(pipe, params_grid, cv=skf, n_jobs=-1, verbose=verbosity, n_iter=randomize, random_state=42)
    else:
        cv_knn = GridSearchCV(pipe, params_grid, cv=skf, n_jobs=-1, verbose=verbosity)
        
    cv_knn.fit(X_train, y_train)
    
    knn_params = cv_knn.best_params_
    train_accuracy = cv_knn.best_score_
    knn_results = cv_knn.cv_results_
        
    print(f"KNN accuracy on training data: {train_accuracy}\n"
          f"Best KNN parameters: {knn_params}")
    
    y_pred = cv_knn.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    final_model = cv_knn.best_estimator_
    final_model.fit(X, y)
    
    print(f"KNN accuracy on test data: {test_accuracy}")
    
    return final_model, knn_results

