import logging

from slicing.slicing_function import slicing_function

logger = logging.getLogger(__name__)

@slicing_function()
def slice_base(example):
    return 1

slices = [
    # slice_base,
]

slice_func_dict = {slice.__name__: slice for slice in slices}
