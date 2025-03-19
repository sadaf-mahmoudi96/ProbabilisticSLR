# -*- coding: utf-8 -*-
"""
Bayesian Neural Network (BNN) for Probabilistic SLR Prediction

Author: smahmoudikouhi
"""

import os
import random
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import requests
from io import StringIO
import keras_tuner as kt
import psutil

# GitHub Base URL for fetching datasets
GITHUB_BASE_URL = "https://raw.githubusercontent.com/sadaf-mahmoudi96/ProbabilisticSLR/main/Input/Year_{year}/All_Points_Sample_{round}_Cluster_{cluster}_IPCC.csv"

# Define global parameters
YEARS = ['2030', '2040', '2050', '2060', '2070', '2080', '2090', '2100']
NUM_CLUSTERS = 2
NUM_ROUNDS = 10
FEATURE_NAMES = ["X", "Y", "vlm", "uv", "heat", "salinity", "sst", "mslp", "sp"]
BATCH_SIZE = 200
NUM_EPOCHS = 100


### 📌 DATA LOADING & PROCESSING ###
def fetch_csv_from_github(year, round_num, cluster):
    """Fetch CSV file from GitHub and load into pandas DataFrame."""
    url = GITHUB_BASE_URL.format(year=year, round=round_num, cluster=cluster)
    response = requests.get(url)

    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        print(f"❌ Failed to fetch {url} - Status Code: {response.status_code}")
        return None


def split_train_validation(df):
    """Split dataset into training & validation sets, ensuring correct duplication handling."""
    duplicate_rows = df[df.duplicated(['X_orig', 'Y_orig'], keep=False)]
    split_dataframes = [group for _, group in duplicate_rows.groupby(['X_orig', 'Y_orig'])]

    selected_groups = random.sample(split_dataframes, 2)
    val_dataset = pd.concat(selected_groups, ignore_index=True)
    
    # Remaining data as training
    train_dataset = pd.concat(split_dataframes, ignore_index=True)
    train_dataset = pd.concat([train_dataset, val_dataset, val_dataset]).drop_duplicates(keep=False)

    return train_dataset.drop(columns=['X_orig', 'Y_orig']), val_dataset.drop(columns=['X_orig', 'Y_orig'])


def get_tf_dataset(df, batch_size=1):
    """Convert Pandas DataFrame to TensorFlow Dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((dict(df.iloc[:, :-1]), df.iloc[:, -1]))
    dataset = dataset.map(lambda x, y: (x, tf.cast(y, tf.float32))).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset.shuffle(len(df)).batch(batch_size)


### 📌 BAYESIAN NEURAL NETWORK MODEL ###
def prior(kernel_size, bias_size, dtype=None):
    """Define Gaussian Prior for Bayesian Neural Network."""
    n = kernel_size + bias_size
    return keras.Sequential([
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n))
        )
    ])


def posterior(kernel_size, bias_size, dtype=None):
    """Define Learnable Posterior for Bayesian Neural Network."""
    n = kernel_size + bias_size
    return keras.Sequential([
        tfp.layers.VariableLayer(tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfp.layers.MultivariateNormalTriL(n),
    ])


def create_bnn_model(hp):
    """Construct Bayesian Neural Network Model with Hyperparameter Tuning."""
    inputs = {feat: layers.Input(name=feat, shape=(1,), dtype=tf.float32) for feat in FEATURE_NAMES}
    features = keras.layers.concatenate(list(inputs.values()))

    for i in range(hp.Int('num_layers', 1, 2)):
        features = tfp.layers.DenseVariational(
            units=hp.Int(f'units_{i}', min_value=32, max_value=320, step=32),
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / 1000,  # Adjust based on dataset size
            activation=hp.Choice('activation', ['relu', 'tanh']),
        )(features)

    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='mse',
        metrics=[keras.metrics.RootMeanSquaredError()]
    )
    return model


### 📌 TRAINING & PREDICTION ###
def run_experiment(model, train_dataset, test_dataset):
    """Train Bayesian Neural Network Model."""
    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.001), loss='mse',
                  metrics=[keras.metrics.RootMeanSquaredError()])
    
    history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset,
                        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    return history


def compute_predictions(model, dataset, iterations=100):
    """Generate Bayesian Predictions with Uncertainty Quantification."""
    examples, targets = list(dataset.unbatch().shuffle(10).batch(len(dataset)))[0]
    predictions = np.concatenate([model(examples).numpy() for _ in range(iterations)], axis=1)

    df_bnn = pd.DataFrame(examples)
    df_bnn['Mean_SLR'] = np.mean(predictions, axis=1)
    df_bnn['Min_SLR'] = np.min(predictions, axis=1)
    df_bnn['Max_SLR'] = np.max(predictions, axis=1)
    df_bnn['Range_SLR'] = df_bnn['Max_SLR'] - df_bnn['Min_SLR']
    df_bnn['Actual_SLR'] = targets

    for p in [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]:
        df_bnn[f'SLR_{p}_percentile'] = np.percentile(predictions, p, axis=1)

    return df_bnn


### 📌 MAIN FUNCTION ###
def main():
    for year in YEARS:
        for cluster in range(NUM_CLUSTERS):
            for round_num in range(NUM_ROUNDS):
                df_all = fetch_csv_from_github(year, round_num, cluster)
                if df_all is None:
                    continue
                
                train_df, val_df = split_train_validation(df_all)

                train_dataset = get_tf_dataset(train_df, batch_size=BATCH_SIZE)
                val_dataset = get_tf_dataset(val_df, batch_size=BATCH_SIZE)

                tuner = kt.Hyperband(create_bnn_model, objective='val_root_mean_squared_error', max_epochs=1, hyperband_iterations=30)
                tuner.search(train_dataset, epochs=1, validation_data=val_dataset, callbacks=[keras.callbacks.EarlyStopping(patience=10)])
                
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                model = tuner.hypermodel.build(best_hps)
                run_experiment(model, train_dataset, val_dataset)

                df_bnn = compute_predictions(model, val_dataset)

                os.makedirs(f"Output/Year_{year}/Cluster_{cluster}/Round_{round_num}", exist_ok=True)
                df_bnn.to_csv(f"Output/Year_{year}/Cluster_{cluster}/Round_{round_num}/BNN_Results.csv", index=False)


if __name__ == "__main__":
    main()
