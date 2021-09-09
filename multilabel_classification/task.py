from functools import partial

from sklearn.metrics import f1_score
from torch import nn
from transformers import AutoModel

from emmental.scorer import Scorer
from emmental.task import EmmentalTask

criterion = nn.BCEWithLogitsLoss()


class FeatureExtractor(nn.Module):
    def __init__(self, feature_extractor):
        super().__init__()

        self.feature_extractor = feature_extractor

    def forward(self, ids):
        outputs = self.feature_extractor(ids)
        return outputs[0][:, 0, :]


def ce_loss(module_name, immediate_ouput_dict, Y, active):
    return criterion(immediate_ouput_dict[module_name][0], Y)


def output(module_name, immediate_ouput_dict):
    return immediate_ouput_dict[module_name][0].sigmoid()


def multi_label_scorer(label_fields, golds, probs, preds, uids):
    acc, f1 = [], []
    res = {}
    for i, key in enumerate(label_fields):
        g = golds[:, i]
        p = probs[:, i] > 0.5
        acc.append((g == p).sum() / len(g))
        f1.append(f1_score(g, p, average="binary"))
        res[f"{key}_accuracy"] = acc[-1]
        res[f"{key}_f1"] = f1[-1]
    res.update({"accuracy": sum(acc) / len(acc), "f1": sum(f1) / len(f1)})

    return res


def create_task(args):
    feature_extractor = AutoModel.from_pretrained(args.model)
    task = EmmentalTask(
        name=args.task_name,
        module_pool=nn.ModuleDict(
            {
                "feature_extractor": FeatureExtractor(feature_extractor),
                "pred_head": nn.Linear(
                    feature_extractor.config.hidden_size, len(args.label_fields)
                ),
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
        scorer=Scorer(
            customize_metric_funcs={
                "multi_label_scorer": partial(multi_label_scorer, args.label_fields)
            }
        ),
    )
    return task
