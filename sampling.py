import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN


def random_undersampling(df, target_col):
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
    # Separate features and target
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Convert the resampled arrays back to a DataFrame
    resampled_df = pd.DataFrame(
        X_resampled, columns=df.drop(target_col, axis=1).columns
    )
    resampled_df[target_col] = y_resampled

    return resampled_df


def perform_adasyn(df, target_col):
    # Separate features and target
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Apply ADASYN
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    # Convert the resampled arrays back to a DataFrame
    resampled_df = pd.DataFrame(
        X_resampled, columns=df.drop(target_col, axis=1).columns
    )
    resampled_df[target_col] = y_resampled

    return resampled_df
