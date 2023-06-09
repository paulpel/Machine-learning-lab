from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler


def random_forest(train_dfs, test_dfs, target_column="Class"):
    """
    Train and evaluate a Random Forest model on multiple datasets.

    :param train_dfs: list of DataFrame
        The list of training datasets.
    :param test_dfs: list of DataFrame
        The list of testing datasets.
    :param target_column: str, optional
        The name of the target column (default is "Class").
    :return: list of dict
        The list of evaluation metrics for each dataset.
    """
    # Separate the features and the targets
    X_train_dfs = [df.drop(target_column, axis=1) for df in train_dfs]
    y_train_dfs = [
        df[target_column].replace({"positive": 1, "negative": 0}) for df in train_dfs
    ]

    X_test_dfs = [df.drop(target_column, axis=1) for df in test_dfs]
    y_test_dfs = [
        df[target_column].replace({"positive": 1, "negative": 0}) for df in test_dfs
    ]

    results = []

    for X_train, y_train, X_test, y_test in zip(
        X_train_dfs, y_train_dfs, X_test_dfs, y_test_dfs
    ):
        # Standardize the features
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Train the model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )  # Set zero_division parameter
        recall = recall_score(
            y_test, predictions, zero_division=0
        )  # Do the same for recall
        f1 = f1_score(y_test, predictions, zero_division=0)  # And for f1 score
        classification_error = 1 - accuracy
        auc_roc = roc_auc_score(y_test, predictions)

        results.append(
            {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "classification_error": classification_error,
                "auc_roc": auc_roc,
            }
        )

    return results


def decision_tree(train_dfs, test_dfs, target_column="Class"):
    """
    Train and evaluate a Decision Tree model on multiple datasets.

    :param train_dfs: list of DataFrame
        The list of training datasets.
    :param test_dfs: list of DataFrame
        The list of testing datasets.
    :param target_column: str, optional
        The name of the target column (default is "Class").
    :return: list of dict
        The list of evaluation metrics for each dataset.
    """
    # Separate the features and the targets
    X_train_dfs = [df.drop(target_column, axis=1) for df in train_dfs]
    y_train_dfs = [
        df[target_column].replace({"positive": 1, "negative": 0}) for df in train_dfs
    ]

    X_test_dfs = [df.drop(target_column, axis=1) for df in test_dfs]
    y_test_dfs = [
        df[target_column].replace({"positive": 1, "negative": 0}) for df in test_dfs
    ]

    results = []

    for X_train, y_train, X_test, y_test in zip(
        X_train_dfs, y_train_dfs, X_test_dfs, y_test_dfs
    ):
        # Standardize the features
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Train the model
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        classification_error = 1 - accuracy
        auc_roc = roc_auc_score(y_test, predictions)

        results.append(
            {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "classification_error": classification_error,
                "auc_roc": auc_roc,
            }
        )

    return results


def naive_bayes(train_dfs, test_dfs, target_column="Class"):
    """
    Train and evaluate a Naive Bayes model on multiple datasets.

    :param train_dfs: list of DataFrame
        The list of training datasets.
    :param test_dfs: list of DataFrame
        The list of testing datasets.
    :param target_column: str, optional
        The name of the target column (default is "Class").
    :return: list of dict
        The list of evaluation metrics for each dataset.
    """
    # Separate the features and the targets
    X_train_dfs = [df.drop(target_column, axis=1) for df in train_dfs]
    y_train_dfs = [
        df[target_column].replace({"positive": 1, "negative": 0}) for df in train_dfs
    ]

    X_test_dfs = [df.drop(target_column, axis=1) for df in test_dfs]
    y_test_dfs = [
        df[target_column].replace({"positive": 1, "negative": 0}) for df in test_dfs
    ]

    results = []

    for X_train, y_train, X_test, y_test in zip(
        X_train_dfs, y_train_dfs, X_test_dfs, y_test_dfs
    ):
        # Standardize the features
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Train the model
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        balanced_accuracy = balanced_accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        classification_error = 1 - accuracy
        auc_roc = roc_auc_score(y_test, predictions)

        results.append(
            {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "classification_error": classification_error,
                "auc_roc": auc_roc,
            }
        )

    return results
