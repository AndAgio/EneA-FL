from .log import get_logger, DumbLogger
from .gpu import get_free_gpu
from .behaviours import read_device_behaviours, get_average_energy, compute_avg_std_time_per_sample
from .dataset import tot_samples_dataset
from .flops import compute_total_number_of_flops
from .optim import get_lr
