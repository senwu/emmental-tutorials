from . import cb, copa, multirc, rte, wic, wsc, swag

parser = {
    "MultiRC": multirc.parse,
    "WiC": wic.parse,
    "CB": cb.parse,
    "COPA": copa.parse,
    "RTE": rte.parse,
    "WSC": wsc.parse,
    "SWAG": swag.parse,
}
