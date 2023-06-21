import json
import pandas as pd
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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

    df = df[df['metric'] == 'balanced_accuracy']

    # Perform one-way ANOVA
    formula = 'value ~ C(model) + C(method) + C(model):C(method)'
    model = smf.ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # Perform Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(df['value'], df['model'] + df['method'])
    print(tukey_result)

    # Export ANOVA table and Tukey's HSD test results to separate files
    anova_table.to_excel("anova_results.xlsx", index=False)
    tukey_result_frame = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
    tukey_result_frame.to_csv("tukey_results.csv", index=False)
