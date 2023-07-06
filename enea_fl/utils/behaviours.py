import os
import numpy as np
import pandas as pd

from .dataset import tot_samples_dataset


def read_device_behaviours(device_type='nano', dataset='femnist'):
    parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    reports_dir = os.path.join(parent_path, 'reports')
    file_name = os.path.join(reports_dir, '{}_{}_5epoch.csv'.format(device_type,
                                                                    dataset))
    behaviour_data = pd.read_csv(file_name)
    return behaviour_data


def average_behaviours(device_behaviour_files):
    energy_records = []
    for file in device_behaviour_files:
        behaviour_data = pd.read_csv(file)
        energy_records.append(behaviour_data['energon_total_in_power_mW'])
    avg_energy_records = np.mean(np.asarray(energy_records), axis=1)
    std_energy_records = np.std(np.asarray(energy_records), axis=1)
    return pd.DataFrame([avg_energy_records, std_energy_records], columns=['energon_total_in_power_mW_avg',
                                                                           'energon_total_in_power_mW_std'])


def compute_avg_std_time_per_sample(device_behaviour_files, dataset='femnist'):
    tot_samples_dataset(dataset=dataset)
    times = []
    for file in device_behaviour_files:
        behaviour_data = pd.read_csv(file)
        times.append(behaviour_data['timestamp'][-1] - behaviour_data['timestamp'][0])
    return np.mean(times), np.std(times)
