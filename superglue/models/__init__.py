from . import cb, copa, multirc, rte, semcor, wic, wsc

model = {
    "MultiRC": multirc.build_model,
    "WiC": wic.build_model,
    "CB": cb.build_model,
    "RTE": rte.build_model,
    "COPA": copa.build_model,
    "WSC": wsc.build_model,
    "SWAG": swag.build_model,
    "SemCor": semcor.build_model,
}
