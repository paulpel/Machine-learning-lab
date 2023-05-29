import os
import json
from load_data import get_paths, load_files_into_dataframes
from sampling import (
    random_undersampling,
    random_oversampling,
    perform_smote,
    perform_adasyn,
)
from handle_pickle import save_pickle, load_pickle
from models import random_forest, decision_tree, naive_bayes
from umce import create_imbalanced_ensemble


class MachineLearning:
    """
    Class for running Machine Learning experiments.

    :param reload_data: bool, optional
        Whether to reload data from the source files (default is False).
    :param perform_sampling: bool, optional
        Whether to perform sampling on the datasets (default is False).
    """

    def __init__(self):
        self.reload_data = False
        self.perform_sampling = False
        self.functions = [
            random_undersampling,
            random_oversampling,
            perform_smote,
            perform_adasyn,
        ]
        self.function_names = [func.__name__ for func in self.functions]

    def main(self):
        """
        Main function to run the machine learning experiments.
        """
        dfs, sampled_dfs = self.load()
        dfs, sampled_dfs = self.prep_data(dfs, sampled_dfs)
        for dataset_name, train_test in dfs.items():
            metrics_rf, metrics_dt, metrics_nb = create_imbalanced_ensemble(
                train_test[0], train_test[1]
            )
            break
        # results_raw = {}
        # for dataset_name, train_test in dfs.items():
        #     dict_temp = {
        #         "random_forest": random_forest(train_test[0], train_test[1]),
        #         "decision_tree": decision_tree(train_test[0], train_test[1]),
        #         "naive_bayes": naive_bayes(train_test[0], train_test[1]),
        #     }
        #     results_raw[dataset_name] = dict_temp
        # self.save_json_results(
        #     "raw_data",
        #     results_raw
        # )

    def prep_data(self, dfs, sampled_dfs):
        """
        Prepare the datasets for the machine learning experiments.

        :param dfs: dict
            Dictionary containing the raw datasets.
        :param sampled_dfs: list
            List of dictionaries containing the sampled datasets.
        :return: tuple
            Tuple containing the prepared raw and sampled datasets.
        """
        new_d = {k: [v[i::2] for i in range(2)] for k, v in dfs.items()}
        new_sampled = []
        for item in sampled_dfs:
            temp = {k: [v[i::2] for i in range(2)] for k, v in item.items()}
            new_sampled.append(temp)

        return new_d, new_sampled

    def save_json_results(self, filename, data):
        """
        Save the results of the experiments as a JSON file.

        :param filename: str
            The name of the JSON file.
        :param data: dict
            The results data to be saved.
        """
        result_path = os.path.join(os.getcwd(), "results")
        path = os.path.join(result_path, filename + ".json")

        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def load(self):
        """
        Load raw and sampled datasets.

        :return: tuple
            Tuple containing raw and sampled datasets.
        """
        cwd = os.getcwd()
        df_path = os.path.join(cwd, "dataframes")
        data_path = os.path.join(df_path, "data.pkl")
        pre_sample, post_sample = "data_", ".pkl"

        if self.reload_data:
            raw_file_paths = get_paths()
            dfs = load_files_into_dataframes(raw_file_paths)
            save_pickle(dfs, data_path)
        else:
            dfs = load_pickle(data_path)

        if self.perform_sampling:
            sampled_dfs = []
            for func in self.functions:
                updated_structure = {}
                for name, dataframes in dfs.items():
                    try:
                        updated_dataframes = [func(df, "Class") for df in dataframes]
                        updated_structure[name] = updated_dataframes
                    except Exception as err:
                        print(err)
                        print(name, func)
                sampled_dfs.append(updated_structure)
            for f_name, data in zip(self.function_names, sampled_dfs):
                save_pickle(
                    data, os.path.join(df_path, pre_sample + f_name + post_sample)
                )
        else:
            sampled_dfs = []
            for f_name in self.function_names:
                sampled_dfs.append(
                    load_pickle(
                        os.path.join(df_path, pre_sample + f_name + post_sample)
                    )
                )

        return dfs, sampled_dfs


if __name__ == "__main__":
    obj = MachineLearning()
    obj.main()
