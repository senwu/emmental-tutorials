import torch.nn.functional as F


def ce_loss(module_name, immediate_ouput_dict, Y, active):
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active]
    )


def ce_loss_multiple_choice(module_name, num_choices, immediate_ouput_dict, Y, active):
    batch_size, dim = immediate_ouput_dict[module_name][0].size()
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0].view(batch_size // num_choices, -1)[
            active
        ],
        (Y.view(-1) - 1)[active],
    )


def output(module_name, immediate_ouput_dict):
    return F.softmax(immediate_ouput_dict[module_name][0], dim=1)


def output_multiple_choice(module_name, num_choices, immediate_ouput_dict):
    batch_size, dim = immediate_ouput_dict[module_name][0].size()
    return F.softmax(
        immediate_ouput_dict[module_name][0].view(batch_size // num_choices, -1), dim=1
    )
