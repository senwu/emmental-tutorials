from functools import partial

from emmental.scorer import Scorer
from emmental.task import EmmentalTask
from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModel


class Feature(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()

        self.feature_extractor = feature_extractor

    def forward(self, ids):
        outputs = self.feature_extractor(ids)
        # import pdb; pdb.set_trace()
        return outputs[0][:, 0, :]


criterion = nn.BCEWithLogitsLoss()


def ce_loss(module_name, immediate_ouput_dict, Y, active):
    # import pdb; pdb.set_trace()
    return criterion(immediate_ouput_dict[module_name][0], Y)


def output(module_name, immediate_ouput_dict):
    return immediate_ouput_dict[module_name][0].sigmoid()


def multif1(golds, probs, preds, uids):
    label_keys = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    # import pdb; pdb.set_trace()
    acc = []
    tot = []
    f1 = []
    res = {}
    for i, key in enumerate(label_keys):
        g = golds[:, i]
        p = probs[:, i] > 0.5
        acc.append((g == p).sum())
        tot.append(len(g))
        f1.append(f1_score(g, p, average="binary"))
        res[f"{key}_accuracy"] = acc[-1] / tot[-1]
        res[f"{key}_f1"] = f1[-1]
    res.update({"accuracy": sum(acc) / sum(tot), "f1": sum(f1) / len(f1)})
    print(f1)
    return res


def create_task(args):
    # Load pretrained transformers
    feature_extractor = AutoModel.from_pretrained(args.model)

    task = EmmentalTask(
        name="toxic",
        module_pool=nn.ModuleDict(
            {
                "feature_extractor": Feature(feature_extractor),
                "pred_head": nn.Linear(feature_extractor.config.hidden_size, 6),
            }
        ),
        task_flow=[
            {
                "name": "feature_extractor",
                "module": "feature_extractor",
                "inputs": [
                    ("_input_", "feat_input_ids"),
                ],
            },
            {
                "name": "pred_head",
                "module": "pred_head",
                "inputs": [("feature_extractor", 0)],
            },
        ],
        loss_func=partial(ce_loss, "pred_head"),
        output_func=partial(output, "pred_head"),
        scorer=Scorer(customize_metric_funcs={"multif1": multif1}),
    )
    return task
