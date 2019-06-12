import logging

from slicing.slicing_function import slicing_function

logger = logging.getLogger(__name__)

@slicing_function()
def slice_base(example):
    return 1

@slicing_function(fields=["sentence2"])
def slice_result(example):
    """Is the Q 'What happened as a result?' (not 'What was the cause of this?')"""
    return example.sentence2 == "What happened as a result?"

slices = [
    slice_base,
]

slice_func_dict = {slice.__name__: slice for slice in slices}
