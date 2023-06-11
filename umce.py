import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    classification_error = 1 - accuracy
    auc_roc = roc_auc_score(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_error": classification_error,
        "auc_roc": auc_roc,
    }


def majority_vote(prediction_lists):
    return mode(prediction_lists, axis=0, keepdims=False)[0][0]


def create_imbalanced_ensemble(train_dfs, test_dfs):
    metrics_rf = []
    metrics_dt = []
    metrics_nb = []

    for train_df, test_df in zip(train_dfs, test_dfs):
        train_df["Class"] = train_df["Class"].map({"negative": 0, "positive": 1})
        test_df["Class"] = test_df["Class"].map({"negative": 0, "positive": 1})
        class_counts = train_df["Class"].value_counts()
        minority_label = class_counts.idxmin()
        majority_label = class_counts.idxmax()

        minority_samples = train_df[train_df["Class"] == minority_label]
        majority_samples = train_df[train_df["Class"] == majority_label]

        imbalance_ratio = len(majority_samples) / len(minority_samples)
        k = round(imbalance_ratio)

        shuffled_majority_samples = majority_samples.sample(frac=1, random_state=42)
        majority_subsets = np.array_split(shuffled_majority_samples, k)

        rf_predictions = []
        dt_predictions = []
        nb_predictions = []

        for i in range(k):
            training_set = pd.concat([majority_subsets[i], minority_samples])

            X_train = training_set.drop("Class", axis=1).values
            y_train = training_set["Class"].values

            X_test = test_df.drop("Class", axis=1).values
            y_test = test_df["Class"].values

            rf_classifier = RandomForestClassifier()  # Random Forest classifier
            rf_classifier.fit(X_train, y_train)
            rf_predictions.append(rf_classifier.predict(X_test))

            dt_classifier = DecisionTreeClassifier()  # Decision Tree classifier
            dt_classifier.fit(X_train, y_train)
            dt_predictions.append(dt_classifier.predict(X_test))

            nb_classifier = GaussianNB()  # Naive Bayes classifier
            nb_classifier.fit(X_train, y_train)
            nb_predictions.append(nb_classifier.predict(X_test))

        # Majority vote
        rf_majority_vote = majority_vote(rf_predictions)
        dt_majority_vote = majority_vote(dt_predictions)
        nb_majority_vote = majority_vote(nb_predictions)

        # Calculate metrics
        metrics_rf.append(calculate_metrics(y_test, rf_majority_vote))
        metrics_dt.append(calculate_metrics(y_test, dt_majority_vote))
        metrics_nb.append(calculate_metrics(y_test, nb_majority_vote))

    return metrics_rf, metrics_dt, metrics_nb
