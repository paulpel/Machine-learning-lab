from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def create_imbalanced_ensemble(train_dfs, test_dfs):
    rf_ensemble = []
    dt_ensemble = []
    nb_ensemble = []
    metrics_rf = []
    metrics_dt = []
    metrics_nb = []

    for train_df, test_df in zip(train_dfs, test_dfs):
        # Step 1: Identify the minority and majority classes
        class_counts = train_df['Class'].value_counts()
        minority_label = class_counts.idxmin()
        majority_label = class_counts.idxmax()

        # Step 2: Divide the dataset into subsets of the minority class (M_inC) and the majority class (M_ajC)
        minority_samples = train_df[train_df['Class'] == minority_label]
        majority_samples = train_df[train_df['Class'] == majority_label]

        # Step 3: Calculate the imbalanced ratio (IR)
        imbalance_ratio = len(majority_samples) / len(minority_samples)

        # Step 4: Round IR to the nearest integer to establish the value of k
        k = round(imbalance_ratio)

        # Step 5: Perform a shuffled k-fold division of M_ajC
        shuffled_majority_samples = majority_samples.sample(frac=1, random_state=42)  # Shuffle the majority samples
        majority_subsets = np.array_split(shuffled_majority_samples, k)

        # Step 6: For each i in the range from 1 to k
        for i in range(k):
            # Step 7: Combine M_ajCi with M_inC to prepare a training set, TSi
            training_set = pd.concat([majority_subsets[i], minority_samples])

            # Step 8: Train classifiers on the training set TSi and add them to the ensembles
            X_train = training_set.drop('Class', axis=1).values
            y_train = training_set['Class'].values

            rf_classifier = RandomForestClassifier()  # Random Forest classifier
            rf_classifier.fit(X_train, y_train)
            rf_ensemble.append(rf_classifier)
            rf_predictions = rf_classifier.predict(X_train)
            rf_metrics = calculate_metrics(y_train, rf_predictions)
            metrics_rf.append(rf_metrics)

            dt_classifier = DecisionTreeClassifier()  # Decision Tree classifier
            dt_classifier.fit(X_train, y_train)
            dt_ensemble.append(dt_classifier)
            dt_predictions = dt_classifier.predict(X_train)
            dt_metrics = calculate_metrics(y_train, dt_predictions)
            metrics_dt.append(dt_metrics)

            nb_classifier = GaussianNB()  # Naive Bayes classifier
            nb_classifier.fit(X_train, y_train)
            nb_ensemble.append(nb_classifier)
            nb_predictions = nb_classifier.predict(X_train)
            nb_metrics = calculate_metrics(y_train, nb_predictions)
            metrics_nb.append(nb_metrics)

    # Return the ensembles of classifiers and the list of metrics
    return metrics_rf, metrics_dt, metrics_nb


def calculate_metrics(y_true, y_pred):
    # Map string labels to numeric values
    label_map = {'negative': 0, 'positive': 1}
    y_true_numeric = [label_map[label] for label in y_true]
    y_pred_numeric = [label_map[label] for label in y_pred]

    accuracy = accuracy_score(y_true_numeric, y_pred_numeric)
    balanced_accuracy = balanced_accuracy_score(y_true_numeric, y_pred_numeric)
    precision = precision_score(y_true_numeric, y_pred_numeric)
    recall = recall_score(y_true_numeric, y_pred_numeric)
    f1 = f1_score(y_true_numeric, y_pred_numeric)
    classification_error = 1 - accuracy
    auc_roc = roc_auc_score(y_true_numeric, y_pred_numeric)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_error": classification_error,
        "auc_roc": auc_roc,
    }

