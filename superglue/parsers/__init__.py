# import multirc_parser
from . import cb, copa, multirc, rte, wic

parser = {
    "MultiRC": multirc.parse,
    "WiC": wic.parse,
    "CB": cb.parse,
    "COPA": copa.parse,
    "RTE": rte.parse,
}
