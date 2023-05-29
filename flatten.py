import json
import pandas as pd


def load_data(file_path):
    # load your data from a json file
    with open(file_path, "r") as f:
        data = json.load(f)

    # flatten json structure and convert to pandas DataFrame
    rows = []
    for dataset, models in data.items():
        for model, runs in models.items():
            for run in runs:
                row = run.copy()
                row["dataset"] = dataset
                row["model"] = model
                rows.append(row)

    df = pd.DataFrame(rows)
    return df
