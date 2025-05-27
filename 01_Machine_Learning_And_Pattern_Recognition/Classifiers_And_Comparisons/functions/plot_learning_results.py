# plot_learning_results.py

import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

'''This function takes the cv_results from a gridsearchCV or a randomizedSearchCV'''
def plot_cv_results(cv_results):
    num_models = len(cv_results)
    nrows = 1 # int(np.ceil(np.sqrt(num_models)))
    ncols = num_models # int(np.ceil(num_models/nrows))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 3*nrows))
    axes = axes.flatten()
    for axis, (model_name, cv_result) in zip(axes, cv_results.items()):
        mean_test_score = cv_result['mean_test_score']
        # std_test_score = cv_result['std_test_score']
    
        axis.plot(range(1, len(mean_test_score) + 1), mean_test_score, marker='o', linestyle='--', label=f"{model_name} Mean Test Score")
        # plt.fill_between(range(1, len(mean_test_score) + 1), 
        #                 mean_test_score - std_test_score, 
        #                 mean_test_score + std_test_score, 
        #                 alpha=0.1, color="r", label='Standard Deviation')
        print(f"{model_name} plot computed")
        axis.set_title('CV Scores')
        axis.set_xlabel('Combinations of Parameters')
        axis.set_ylabel('Accuracy')
        axis.set_ylim(0, 1)
        axis.legend(loc='best')
        axis.grid(True)
    
    for i in range(len(cv_results), len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()

'''This function takes a dict of trained models and their cv as well as related data set as values and plots their learning curve'''
def plot_learning_curve(models, y, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)):
    fig, axes = plt.subplots(1, 2, figsize=(20,8))
    for model_name, (model, cv, X) in models.items():
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=3)
        
        train_scores_mean = np.mean(train_scores, axis=1)
        # train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        # test_scores_std = np.std(test_scores, axis=1)
        
        # axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1)
        # axes[1].fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)
        
        axes[0].plot(train_sizes, train_scores_mean, 'o-', label=f"{model_name} Training")
        axes[1].plot(train_sizes, test_scores_mean, 'o-', label=f"{model_name} CV")
        print(f"{model_name} plot computed")
    
    axes[0].set_title('Score on Training Data')
    axes[0].set_xlabel('Training Size')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0.4, 1.05)
    axes[0].legend(loc='best') # , bbox_to_anchor=(1, 1)
    axes[0].grid(True)
    axes[1].set_title('Score on Cross-Validated Data')
    axes[1].set_xlabel('Training Size')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0.4, 1.05)
    axes[1].legend(loc='best') # , bbox_to_anchor=(1, 1) for label outside
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()