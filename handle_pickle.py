import pickle


def save_pickle(obj, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(file_path):
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    return obj
