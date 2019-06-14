from . import WiC_afs

# sys.path.append("..")  # Adds higher directory to python modules path.

augmentation_funcs = {
    # "CB": CB_afs.augmentation_funcs,
    # "COPA": COPA_afs.augmentation_funcs,
    # "MultiRC": MultiRC_afs.augmentation_funcs,
    # "RTE": RTE_afs.augmentation_funcs,
    "WiC": WiC_afs.augmentation_funcs,
    # "WSC": WSC_afs.augmentation_funcs,
}
