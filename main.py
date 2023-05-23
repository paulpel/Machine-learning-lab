import os
from load_data import get_paths, load_files_into_dataframes
from sampling import (
    random_undersampling,
    random_oversampling,
    perform_smote,
    perform_adasyn,
)
from handle_pickle import save_pickle, load_pickle


if __name__ == "__main__":
    reload_data = True
    perform_sampling = True

    functions = [
        random_undersampling,
        random_oversampling,
        perform_smote,
        perform_adasyn,
    ]
    function_names = [func.__name__ for func in functions]

    cwd = os.getcwd()
    df_path = os.path.join(cwd, "dataframes")
    data_path = os.path.join(df_path, "data.pkl")
    pre_sample, post_sample = "data_", ".pkl"

    if reload_data:
        raw_file_paths = get_paths()
        dfs = load_files_into_dataframes(raw_file_paths)
        save_pickle(dfs, data_path)
    else:
        dfs = load_pickle(data_path)

    if perform_sampling:
        sampled_dfs = []
        for func in functions:
            updated_structure = {}
            for name, dataframes in dfs.items():
                try:
                    updated_dataframes = [func(df, "Class") for df in dataframes]
                    updated_structure[name] = updated_dataframes
                except Exception as err:
                    print(err)
                    print(name, func)
            sampled_dfs.append(updated_structure)
        for f_name, data in zip(function_names, sampled_dfs):
            save_pickle(data, os.path.join(df_path, pre_sample + f_name + post_sample))
    else:
        sampled_dfs = []
        for f_name in function_names:
            sampled_dfs.append(
                load_pickle(os.path.join(df_path, pre_sample + f_name + post_sample))
            )
