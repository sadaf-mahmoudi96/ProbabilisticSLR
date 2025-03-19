# -*- coding: utf-8 -*-
"""
Script for Probabilistic Sea Level Rise (SLR) Prediction using 
Random Forest Quantile Regression.

Author: smahmoudikouhi
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from quantile_forest import RandomForestQuantileRegressor
import random
import requests
from io import StringIO
import os

# Define Parameters
YEARS = ['2030', '2040', '2050', '2060', '2070', '2080', '2090', '2100']
NUM_CLUSTERS = 2  # Number of clusters
NUM_ROUNDS = 10  # Number of rounds for sampling
NUM_SPLITS = 10  # Number of k-fold cross-validation
SPLIT_SIZE = 2   # Size of each split

# GitHub base URL
GITHUB_BASE_URL = "https://raw.githubusercontent.com/sadaf-mahmoudi96/ProbabilisticSLR/main/Input/Year_{year}/All_Points_Sample_{round}_Cluster_{cluster}_IPCC.csv"

# Define Hyperparameter Grid
HYPERPARAM_GRID = {
    'n_estimators': [60, 80, 100, 200, 300, 400, 500],
    'max_depth': [5, 7, 9, 10]
}


def fetch_csv_from_github(year, round_num, cluster):
    """
    Fetch CSV file from GitHub and load it into a pandas DataFrame.
    """
    url = GITHUB_BASE_URL.format(year=year, round=round_num, cluster=cluster)
    response = requests.get(url)
    
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        print(f"Failed to fetch {url} - Status Code: {response.status_code}")
        return None


def split_train_validation(df):
    """
    Split dataset into training and validation sets while ensuring duplicate rows 
    (based on 'X_orig', 'Y_orig') are handled correctly.
    """
    duplicate_rows = df[df.duplicated(['X_orig', 'Y_orig'], keep=False)]
    split_dataframes = [group for _, group in duplicate_rows.groupby(['X_orig', 'Y_orig'])]

    # Randomly select validation groups
    selected_groups = random.sample(split_dataframes, SPLIT_SIZE)
    val_dataset = pd.concat(selected_groups, ignore_index=True)

    # Remaining groups form the training dataset
    train_dataset = pd.concat(split_dataframes, ignore_index=True)
    train_dataset = pd.concat([train_dataset, val_dataset, val_dataset]).drop_duplicates(keep=False)

    return train_dataset, val_dataset


def train_random_forest(mpg_X_train, mpg_y_train):
    """
    Perform hyperparameter tuning using RandomizedSearchCV and train the 
    Random Forest Quantile Regressor model.
    """
    regressor = RandomForestQuantileRegressor()
    clf = RandomizedSearchCV(regressor, HYPERPARAM_GRID, n_iter=10, verbose=2, 
                             random_state=42, n_jobs=-1)
    search = clf.fit(mpg_X_train.iloc[:, :9], mpg_y_train.iloc[:, 0])
    best_params = search.best_params_

    # Train with best parameters
    model = RandomForestQuantileRegressor(n_estimators=best_params['n_estimators'], 
                                          max_depth=best_params['max_depth'], 
                                          random_state=42)
    model.fit(mpg_X_train.iloc[:, :9], mpg_y_train.iloc[:, 0])
    
    return model


def predict_and_format_results(model, mpg_X, mpg_y):
    """
    Generate predictions for different quantiles and format the results into a DataFrame.
    """
    quantiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                 0.95, 0.99, 0.167, 0.833, 0.995, 0.999]
    
    y_pred = model.predict(mpg_X.iloc[:, :9], quantiles=quantiles)
    
    # Compute prediction intervals
    y_pred_interval = y_pred[:, 11] - y_pred[:, 1]
    sort_idx = np.argsort(y_pred_interval)

    X_sorted = mpg_X.iloc[sort_idx]
    mpg_y = mpg_y.iloc[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    # Combine results into a structured DataFrame
    columns = ["X", "Y", "VLM", "uv", "heat", "salinity", "mslp", "sst", "sp", 'X_orig', 'Y_orig'] + \
              [f"SLR_{int(q*100)}_percentile" for q in quantiles] + ["SLR_Interval", "Actual_SLR"]

    combined_array = np.column_stack((X_sorted, y_pred_sorted, y_pred_interval, mpg_y.iloc[:, 0]))
    df_results = pd.DataFrame(combined_array, columns=columns)
    
    return df_results


def main():
    """
    Main execution loop to process data for different years, clusters, and rounds.
    """
    for year in YEARS:
        for cluster in range(NUM_CLUSTERS):
            for round_num in range(NUM_ROUNDS):
                df_all = fetch_csv_from_github(year, round_num, cluster)
                if df_all is None:
                    continue
                
                train_dataset, val_dataset = split_train_validation(df_all)

                mpg_X_train, mpg_X_test, mpg_y_train, mpg_y_test = train_test_split(
                    train_dataset.iloc[:, :11], train_dataset.iloc[:, -1:], 
                    random_state=42, test_size=0.2, shuffle=True
                )

                # Train model
                model = train_random_forest(mpg_X_train, mpg_y_train)

                # Predict and format results for train, test, and validation sets
                df_train = predict_and_format_results(model, mpg_X_train, mpg_y_train)
                df_test = predict_and_format_results(model, mpg_X_test, mpg_y_test)
                df_val = predict_and_format_results(model, val_dataset.iloc[:, :11], val_dataset.iloc[:, -1:])

                # Save results
                output_dir = f"Output/Year_{year}/Cluster_{cluster}/Round_{round_num}"
                os.makedirs(output_dir, exist_ok=True)

                df_train.to_csv(f"{output_dir}/Train_Results.csv", index=False)
                df_test.to_csv(f"{output_dir}/Test_Results.csv", index=False)
                df_val.to_csv(f"{output_dir}/Validation_Results.csv", index=False)

                print(f"Saved results for Year {year}, Cluster {cluster}, Round {round_num}")


if __name__ == "__main__":
    main()
