# data_cleaning.py

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.experimental import enable_iterative_imputer  # Apparently required to enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

def plot_dist(X_df):   # Plot the distribution of the features (takes numerical features only)
    sns.set_theme()
    features = X_df.columns
    num_features = len(features)
    fig_rows = math.ceil(math.sqrt(num_features))
    fig_cols = math.ceil(num_features/fig_rows)

    sns.pairplot(data=X_df, vars=features)
    plt.show()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(X_df.corr("pearson", numeric_only=True).abs(), annot=True)
    plt.show()
    
    # Also plot a boxplot to identify outliers    
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(15,10))
    fig.tight_layout(pad=2)

    for i, col in enumerate(features):
        if i > num_features:
            break
        sns.boxplot(X_df[col], ax=axes[i // fig_cols, i % fig_cols])
    

def remove_outliers(X_df, scaler=1.5): # Using the inter quartile range method. Apparently better to use when data is kind of normally dist
    X_cleaned = X_df.copy()
    for col in X_df.select_dtypes(include=["float64", "int64"]).columns:
        Q1 = X_df[col].quantile(0.25)
        Q3 = X_df[col].quantile(0.75)
        IQR = Q3 - Q1
        low_bound = Q1 - scaler*IQR
        up_bound = Q3 + scaler*IQR
        X_cleaned[col] = X_df[col].apply(lambda x: np.nan if x < low_bound or x > up_bound else x)
    return X_cleaned

def replace_outliers(X_df, low_quant=0.01, up_quant=0.99): # Using the inter quartile range method. Apparently better to use when data is kind of normally dist
    X_no_outliers = X_df.copy()
    for col in X_df.select_dtypes(include=["float64", "int64"]).columns:
        low_bound = X_df[col].quantile(low_quant)
        up_bound = X_df[col].quantile(up_quant)
        X_no_outliers[col] = X_df[col].apply(lambda x: np.nan if x < low_bound or x > up_bound else x)
    return X_no_outliers

def impute_nans(X_no_outliers):
    imputer = IterativeImputer(random_state=42)
    X_imputed = imputer.fit_transform(X_no_outliers)
    X_imputed = pd.DataFrame(X_imputed, columns=X_no_outliers.columns)
    return X_imputed

def robust_scale(X_df):
    scaler = RobustScaler()
    numerical_features = X_df.select_dtypes(include=["float64", "int64"])
    X_scaled = scaler.fit_transform(numerical_features)
    X_df[numerical_features.columns] = X_scaled
    return X_df

def clean_data(X_df, low_quant=0.01, up_quant=0.99, replace=True):
    if replace:
        X_no_outliers = replace_outliers(X_df, low_quant, up_quant)
    else:
        X_no_outliers = remove_outliers(X_df)
    X_imputed = impute_nans(X_no_outliers)
    X_clean = robust_scale(X_imputed) 
    return X_clean