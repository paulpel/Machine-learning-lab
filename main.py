import os
from load_data import get_paths, load_files_into_dataframes
from sampling import (
    random_undersampling,
    random_oversampling,
    perform_smote,
    perform_adasyn,
)
from handle_pickle import save_pickle, load_pickle
from models import random_forest


class MachineLearning:
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
        dfs, sampled_dfs = self.load()
        dfs, sampled_dfs = self.prep_data(dfs, sampled_dfs)
        count = 1
        for dataset_name, train_test in dfs.items():
            print(count)
            print(dataset_name)
            res = random_forest(train_test[0], train_test[1])
            for i, result in enumerate(res):
                print(f"Fold {i+1} Metrics: {result}")
            count += 1
            print()

    def prep_data(self, dfs, sampled_dfs):
        new_d = {k: [v[i::2] for i in range(2)] for k, v in dfs.items()}
        new_sampled = []
        for item in sampled_dfs:
            temp = {k: [v[i::2] for i in range(2)] for k, v in item.items()}
            new_sampled.append(temp)

        return new_d, new_sampled

    def load(self):
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
