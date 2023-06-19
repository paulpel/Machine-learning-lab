import json
import pandas as pd
import os


def load_data(directory):
    """
    Load data from all json files in a directory that contain "average" in their filename,
    and flatten it to a pandas DataFrame.

    :param directory: str
        Path to the directory containing the json files.
    :return: pandas.DataFrame
        DataFrame containing the loaded data.
    """

    # Find all json files in the directory that contain "average" in the filename
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".json") and "average" in f]

    # List to store data from all files
    all_data = []

    for file_path in file_paths:
        # Load the data from the json file
        with open(file_path, "r") as f:
            data = json.load(f)

        # Method attribute is derived from the file name
        method = os.path.splitext(os.path.basename(file_path))[0]
        method = method.split('_')[-1]

        # Flatten json structure
        rows = []
        for dataset, models in data.items():
            for model, metrics in models.items():
                for metric, value in metrics.items():
                    # Check if the metric is one of the ones we're interested in
                    if metric in {"f1_score", "balanced_accuracy", "auc_roc"}:
                        row = {
                            "dataset": dataset,
                            "model": model,
                            "metric": metric,
                            "value": value,
                            "method": method  # Added method attribute
                        }
                        rows.append(row)

        # Convert the data for this file into a DataFrame and add it to the list
        all_data.append(pd.DataFrame(rows))

    # Concatenate all data into a single DataFrame
    df = pd.concat(all_data, ignore_index=True)
    return df


if __name__ == "__main__":
    df = load_data(os.path.join(os.getcwd(), 'results'))
    df.to_excel("flatten.xlsx", index=False)
    print(df)
