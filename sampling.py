import pandas as pd
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, ADASYN
import numpy as np
from sklearn.neighbors import NearestNeighbors

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

    # Check the number of unique classes in the target column
    num_classes = len(set(y))

    # Determine the number of neighbors for SMOTE
    n_neighbors = min(5, num_classes - 1)  # Set the maximum number of neighbors to (num_classes - 1)

    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Convert the resampled arrays back to a DataFrame
    resampled_df = pd.DataFrame(X_resampled, columns=df.drop(target_col, axis=1).columns)
    resampled_df[target_col] = y_resampled

    return resampled_df


def perform_adasyn(df, target_col):
    # Separate features and target
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Check the number of unique classes in the target column
    num_classes = len(set(y))

    # Determine the number of neighbors for ADASYN
    n_neighbors = min(5, num_classes - 1)  # Set the maximum number of neighbors to (num_classes - 1)

    # Apply ADASYN
    adasyn = ADASYN(random_state=42, n_neighbors=n_neighbors)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)

    # Convert the resampled arrays back to a DataFrame
    resampled_df = pd.DataFrame(X_resampled, columns=df.drop(target_col, axis=1).columns)
    resampled_df[target_col] = y_resampled

    return resampled_df

def perform_enn(df, target_col):
    # Separate features and target
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values

    # Create an instance of the k-NN classifier
    knn = NearestNeighbors(n_neighbors=3)  # Set the number of neighbors to consider

    minority_class_label = min(set(y))  # Assuming minority class is the smallest class

    # Filter the minority class samples
    minority_indices = np.where(y == minority_class_label)[0]
    minority_samples = X[minority_indices]

    # Fit the k-NN classifier on the minority class samples
    knn.fit(minority_samples)

    selected_indices = []

    # Iterate over each minority class sample
    for i, sample in enumerate(minority_samples):
        # Find the indices of the K nearest neighbors
        _, indices = knn.kneighbors([sample])

        # Check if the majority class is the most frequent class among the neighbors
        majority_count = np.sum(y[minority_indices[indices[0]]] != minority_class_label)

        if majority_count < len(indices[0]) / 2:
            selected_indices.append(minority_indices[i])

    # Filter the dataset based on the selected indices
    resampled_df = df.iloc[selected_indices]

    return resampled_df

# Przykładowe użycie
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6],
    'feature2': [2, 3, 4, 5, 6, 7],
    'target': [0, 0, 1, 1, 1, 1]
})

resampled_df = perform_enn(df, 'target')
print(resampled_df)