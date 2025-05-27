# svm_methods.py

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np

def svm_method(X, y, n_splits=3, randomize=200, verbosity=2, test_size=0.2):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    num_features = X.shape[1]
    
    pipe = Pipeline([       # Defining the classifier using principal component analysis to avoid multicollinearity as much as possible
            ('scaler', StandardScaler()),   # Scaling features
            ('select', SelectKBest(score_func=f_classif, k=8)),
            ('pca', PCA(n_components=0.95, random_state=42)),
            ('svm', SVC(random_state=42))
            ])
    k_features = list(np.cumsum(np.ones(num_features)).astype(dtype=int))[6:]
    params_grid = {              # Defining the kernels to use 
            'svm__C': [0.1, 1, 10, 100, 1000],        #Regularization to prevent overfitting
            'svm__gamma': [1, 0.1, 0.001, 1e-3],    # Smoothness of decision boundary
            'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'pca__n_components': [0.9, 0.95, 0.98, 0.99
                                  #'mle'
                                  ],
            'select__k': [6, 8, 10, 11]
            }
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) # Defining the skf
    
    if randomize > 0:
        cv_svm = RandomizedSearchCV(pipe, params_grid, cv=skf, n_jobs=-1, verbose=verbosity, n_iter=randomize, random_state=42)
    else:
        cv_svm = GridSearchCV(pipe, params_grid, cv=skf, n_jobs=-1, verbose=verbosity)
    
    cv_svm.fit(X_train, y_train)
    
    svm_params = cv_svm.best_params_
    train_accuracy = cv_svm.best_score_
    svm_results = cv_svm.cv_results_
        
    print(f"SVM accuracy on training data: {train_accuracy}\n"
          f"Best SVM parameters: {svm_params}")
    
    y_pred = cv_svm.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # final_model = SVC(**svm_params_cleaned, random_state=42)
    final_model = cv_svm.best_estimator_
    final_model.fit(X, y)

    print(f"SVM accuracy on test data: {test_accuracy}")
    
    return final_model, svm_results