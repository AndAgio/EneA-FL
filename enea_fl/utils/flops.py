from torch import nn


def compute_total_number_of_flops(model, batch_size):
    total_flops = 0
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            total_flops += layer.in_features * layer.out_features * batch_size
        elif isinstance(layer, nn.Conv1d):
            total_flops += layer.in_channels * layer.out_channels * layer.kernel_size[0] * batch_size
        elif isinstance(layer, nn.Conv2d):
            total_flops += 2 * layer.in_channels * layer.out_channels * layer.kernel_size[0] * batch_size
    return total_flops
