import pickle


def save_pickle(obj, file_path):
    """
    Save an object to a file in pickle format.

    :param obj: object
        The object to be saved.
    :param file_path: str
        Path to the file where the object will be saved.
    """
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(file_path):
    """
    Load an object from a file in pickle format.

    :param file_path: str
        Path to the file from which the object will be loaded.
    :return: object
        The loaded object.
    """
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
    return obj
