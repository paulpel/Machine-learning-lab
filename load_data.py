import pandas as pd
import os
import arff
import re
import tempfile


def remove_range_specification(arff_file):
    with open(arff_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith("@attribute"):
            line = re.sub(r"\[.*?\]", "", line)
        new_lines.append(line)

    temp = tempfile.NamedTemporaryFile(delete=False)
    with open(temp.name, "w") as f:
        f.writelines(new_lines)

    return temp.name


def load_files_into_dataframes(file_paths):
    dataframes = {}

    for dir_paths in file_paths:
        dir_dataframes = []

        for path in dir_paths:
            temp_path = remove_range_specification(path)
            with open(temp_path, "r") as f:
                data_dict = arff.load(f)

            df = pd.DataFrame(
                data_dict["data"], columns=[i[0] for i in data_dict["attributes"]]
            )
            if "Class" not in df.columns:
                # Rename the specific column
                df = df.rename(columns={"class": "Class"})
            dir_dataframes.append(df)
        dataframes[data_dict["relation"]] = dir_dataframes

    return dataframes


def get_paths():
    main_raw_path = os.path.join(os.getcwd(), "data_raw")

    dir_file_paths = []

    for subdir in os.listdir(main_raw_path):
        subdir_path = os.path.join(main_raw_path, subdir)

        if os.path.isdir(subdir_path):
            subdir_file_paths = []

            for root, directories, files in os.walk(subdir_path):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    subdir_file_paths.append(filepath)

            dir_file_paths.append(subdir_file_paths)

    return dir_file_paths
