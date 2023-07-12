import os
import numpy as np
import pandas as pd
import glob
from scipy.optimize import curve_fit
from scipy.stats import norm

from .dataset import tot_samples_dataset


def read_device_behaviours(device_type='nano_gpu', dataset='femnist'):
    parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    reports_dir = os.path.join(parent_path, 'reports')

    dev_board = device_type.split('_')[0]
    if dev_board in ['nano', 'orin']:
        dev_board = 'jetson_{}'.format(dev_board)
    use_gpu = False if device_type in ['raspberrypi', 'nano_cpu', 'orin_cpu'] else True

    files_dir = os.path.join(reports_dir, dev_board,
                             'gpu' if use_gpu else 'cpu', 'split_dfs')
    files = [file for file in glob.glob(os.path.join(files_dir, '*.csv')) if dataset in file]
    return files


def get_average_energy(device_behaviour_files, dataset_size, dataset='femnist'):
    sizes = ['small', 'medium', 'big']
    energy_records = {size: [] for size in sizes}
    for file in device_behaviour_files:
        size = file.split('/')[-1].split('_')[1]
        behaviour_data = pd.read_csv(file)
        energy_records[size] += np.asarray(behaviour_data['energon_total_in_power_mW']).tolist()
    avg_energy_records = {size: [] for size in sizes}
    std_energy_records = {size: [] for size in sizes}
    for size in sizes:
        mu, std = norm.fit(np.asarray(energy_records[size]))
        avg_energy_records[size] = mu  # np.mean(np.asarray(energy_records[size]))
        std_energy_records[size] = std  # np.std(np.asarray(energy_records[size]))

    def obj_func(x, a, b):
        return a * x + b

    xs = [tot_samples_dataset(dataset=dataset, size=size) for size in sizes]
    ys = list(avg_energy_records.values())
    popt, _ = curve_fit(obj_func, xs, ys)
    a, b = popt
    avg_energy = obj_func(dataset_size, a, b)

    ys = list(std_energy_records.values())
    popt, _ = curve_fit(obj_func, xs, ys)
    a, b = popt
    std_energy = obj_func(dataset_size, a, b)

    return avg_energy, std_energy


def compute_avg_std_time_per_sample(device_behaviour_files, dataset_size, dataset='femnist'):
    sizes = ['small', 'medium', 'big']
    times = {size: [] for size in sizes}
    for file in device_behaviour_files:
        size = file.split('/')[-1].split('_')[1]
        n_samples = tot_samples_dataset(dataset=dataset,
                                        size=size)
        behaviour_data = pd.read_csv(file)
        timestamps = list(behaviour_data['timestamp'])
        times[size].append((timestamps[-1] - timestamps[0])/n_samples)

    avg_times = {size: [] for size in sizes}
    std_times = {size: [] for size in sizes}
    for size in sizes:
        mu, std = norm.fit(np.asarray(times[size]))
        avg_times[size] = mu  # np.mean(np.asarray(times[size]))
        std_times[size] = std  # np.std(np.asarray(times[size]))

    print('avg_times: {}'.format(avg_times))
    print('std_times: {}'.format(std_times))

    def obj_func(x, a, b):
        return a * x + b

    xs = [tot_samples_dataset(dataset=dataset, size=size) for size in sizes]
    ys = list(avg_times.values())
    popt, _ = curve_fit(obj_func, xs, ys)
    a, b = popt
    avg_time = obj_func(dataset_size, a, b)

    ys = list(std_times.values())
    popt, _ = curve_fit(obj_func, xs, ys)
    a, b = popt
    std_time = obj_func(dataset_size, a, b)

    return abs(avg_time), abs(std_time)

