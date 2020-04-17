import json
import os

def write_to_file(file_path, value):
    """
    Write value to file.
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    if not isinstance(value, str):
        value = str(value)
    fout = open(file_path, "w")
    fout.write(value + "\n")
    fout.close()


def write_to_json_file(file_path, dict):
    """
    Write dict to json file.
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    json_obj = json.dumps(dict)
    fout = open(file_path, "w")
    fout.write(json_obj)
    fout.close()
