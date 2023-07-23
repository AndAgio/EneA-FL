import torch
import numpy as np

def get_free_gpu():
    # jetson devices do not have this attribute
    if not hasattr(torch.cuda, "mem_get_info"):
        return 0
    free_memories = []
    for i in range(torch.cuda.device_count()):
        device = torch.device('cuda:{}'.format(i))
        # torch.cuda.mem_get_info
        # Returns the global free and total GPU memory occupied for a given device using cudaMemGetInfo.
        free_memory = torch.cuda.mem_get_info(device)[0]
        free_memories += [free_memory]
    return np.argmax(np.asarray(free_memories))
