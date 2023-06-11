import os
import json
from collections import defaultdict


def average_metrics(json_dict):
    averaged_dict = defaultdict(dict)

    for dataset, models in json_dict.items():
        for model, metrics_list in models.items():
            averaged_metrics = defaultdict(float)

            for metrics in metrics_list:
                for metric, value in metrics.items():
                    averaged_metrics[metric] += value

            for metric, value in averaged_metrics.items():
                averaged_metrics[metric] = value / len(metrics_list)

            averaged_dict[dataset][model] = dict(averaged_metrics)

    return dict(averaged_dict)


def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            with open(os.path.join(directory_path, filename), "r") as file:
                data = json.load(file)

            averaged_data = average_metrics(data)

            with open(os.path.join(directory_path, f"average_{filename}"), "w") as file:
                json.dump(averaged_data, file, indent=4)


if __name__ == "__main__":
    cwd = os.getcwd()
    dir_path = os.path.join(cwd, "results")
    process_directory(dir_path)
