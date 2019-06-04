import os


def str2list(v, dim=","):
    return [t.strip() for t in v.split(dim)]


def write_to_file(path, file_name, value):
    if not isinstance(value, str):
        value = str(value)
    fout = open(os.path.join(path, file_name), "w")
    fout.write(value + "\n")
    fout.close()
