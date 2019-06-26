from . import cb, copa, multirc, rte, semcor, swag, wic, wsc

parser = {
    "MultiRC": multirc.parse,
    "WiC": wic.parse,
    "CB": cb.parse,
    "COPA": copa.parse,
    "RTE": rte.parse,
    "WSC": wsc.parse,
    "SWAG": swag.parse,
    "SemCor": semcor.parse,
}
