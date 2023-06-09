import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN


def random_undersampling(df, target_col):
    """
    Perform random undersampling to balance the classes.

    :param df: DataFrame
        The input dataframe with imbalanced classes.
    :param target_col: str
        The name of the target column (class labels).
    :return: DataFrame
        The resulting dataframe after undersampling.
    """
    # Calculate class frequencies
    class_frequencies = df[target_col].value_counts()

    # Identify majority and minority classes
    majority_class = class_frequencies.idxmax()
    minority_class = class_frequencies.idxmin()

    # Separate the majority and minority classes
    majority_instances = df[df[target_col] == majority_class]
    minority_instances = df[df[target_col] == minority_class]

    # Perform random undersampling on the majority class
    majority_undersampled = resample(
        majority_instances,
        replace=False,
        n_samples=len(minority_instances),
        random_state=42,
    )

    # Concatenate the undersampled majority class with the minority class
    undersampled_df = pd.concat([majority_undersampled, minority_instances])

    return undersampled_df


def random_oversampling(df, target_col):
    """
    Perform random oversampling to balance the classes.

    :param df: DataFrame
        The input dataframe with imbalanced classes.
    :param target_col: str
        The name of the target column (class labels).
    :return: DataFrame
        The resulting dataframe after oversampling.
    """
    # Calculate class frequencies
    class_frequencies = df[target_col].value_counts()

    # Identify majority and minority classes
    majority_class = class_frequencies.idxmax()
    minority_class = class_frequencies.idxmin()

    # Separate the majority and minority classes
    majority_instances = df[df[target_col] == majority_class]
    minority_instances = df[df[target_col] == minority_class]

    # Perform random oversampling on the minority class
    minority_oversampled = resample(
        minority_instances,
        replace=True,
        n_samples=len(majority_instances),
        random_state=42,
    )

    # Concatenate the oversampled minority class with the majority class
    oversampled_df = pd.concat([majority_instances, minority_oversampled])

    return oversampled_df


def perform_smote(df, target_col):
    """
    Perform Synthetic Minority Over-sampling Technique (SMOTE) to balance the classes.

    :param df: DataFrame
        The input dataframe with imbalanced classes.
    :param target_col: str
        The name of the target column (class labels).
    :return: DataFrame
        The resulting dataframe after applying SMOTE.
    """
    # Separate features and target
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Check the number of unique classes in the target column
    num_classes = len(set(y))

    # Determine the number of neighbors for SMOTE
    n_neighbors = min(
        5, num_classes - 1
    )  # Set the maximum number of neighbors to (num_classes - 1)

    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Convert the resampled arrays back to a DataFrame
    resampled_df = pd.DataFrame(
        X_resampled, columns=df.drop(target_col, axis=1).columns
    )
    resampled_df[target_col] = y_resampled

    return resampled_df


def perform_adasyn(df, target_col):
    """
    Perform Adaptive Synthetic (ADASYN) sampling approach to balance the classes.

    :param df: DataFrame
        The input dataframe with imbalanced classes.
    :param target_col: str
        The name of the target column (class labels).
    :return: DataFrame
        The resulting dataframe after applying ADASYN.
    """
    # Separate features and target
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Check the number of unique classes in the target column
    num_classes = len(set(y))

    # Determine the number of neighbors for ADASYN
    n_neighbors = min(
        5, num_classes - 1
    )  # Set the maximum number of neighbors to (num_classes - 1)

    # Apply ADASYN
    adasyn = ADASYN(random_state=42, n_neighbors=n_neighbors)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    # Convert the resampled arrays back to a DataFrame
    resampled_df = pd.DataFrame(
        X_resampled, columns=df.drop(target_col, axis=1).columns
    )
    resampled_df[target_col] = y_resampled

    return resampled_df
