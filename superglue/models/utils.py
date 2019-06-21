import torch.nn.functional as F


def ce_loss(module_name, immediate_ouput_dict, Y, active):
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0][active], (Y.view(-1) - 1)[active]
    )

def ce_loss_new(module_name, immediate_ouput_dict, Y, active):
    import pdb; pdb.set_trace()
    batch_size, dim = immediate_ouput_dict[module_name][0].size()
    return F.cross_entropy(
        immediate_ouput_dict[module_name][0].view(batch_size // 2, -1)[active], (Y.view(-1) - 1)[active]
    )

def output(module_name, immediate_ouput_dict):
    return F.softmax(immediate_ouput_dict[module_name][0], dim=1)

def output_new(module_name, immediate_ouput_dict):
    import pdb; pdb.set_trace()
    batch_size, dim = immediate_ouput_dict[module_name][0].size()
    return F.softmax(immediate_ouput_dict[module_name][0].view(batch_size // 2, -1), dim=1)
