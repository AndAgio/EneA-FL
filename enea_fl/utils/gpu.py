import torch
import numpy as np


def get_free_gpu():
    free_memories = []
    for i in range(torch.cuda.device_count()):
        device = torch.device('cuda:{}'.format(i))
        free_memories += torch.cuda.mem_get_info(device)
    return np.argmax(np.asarray(free_memories))
