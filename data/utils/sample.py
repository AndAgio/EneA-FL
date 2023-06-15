'''
samples from all raw data;
by default samples in a non-iid manner; namely, randomly selects users from 
raw data until their cumulative amount of data exceeds the given number of 
datapoints to sample (specified by --fraction argument);
ordering of original data points is not preserved in sampled data
'''

import argparse
import json
import os
import random
import time
import numpy as np
from scipy.special import softmax

from collections import OrderedDict

from constants import DATASETS, SEED_FILES, SAMPLE_MODES

parser = argparse.ArgumentParser()

parser.add_argument('--name',
                    help='name of dataset to parse; default: sent140;',
                    type=str,
                    choices=DATASETS,
                    default='sent140')
parser.add_argument('--sampling_mode',
                    help='define how samples are distributed amongst workers; default: iid+sim;',
                    type=str,
                    choices=SAMPLE_MODES,
                    default='iid+sim')
parser.add_argument('--fraction',
                    help='fraction of all data to sample; default: 0.1;',
                    type=float,
                    default=0.1)
parser.add_argument('--u',
                    help='number of workers to consider;',
                    type=int,
                    default=100)
parser.add_argument('--spw',
                    help='maximum number of samples for each worker;',
                    type=int,
                    default=10000)
parser.add_argument('--seed',
                    help='seed for random sampling of data',
                    type=int,
                    default=None)
parser.set_defaults(iid=False)

args = parser.parse_args()

print('------------------------------')
print('sampling data')

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
data_dir = os.path.join(parent_path, args.name, 'data')
subdir = os.path.join(data_dir, 'all_data')
files = os.listdir(subdir)
files = [f for f in files if f.endswith('.json')]

rng_seed = (args.seed if (args.seed is not None and args.seed >= 0) else int(time.time()))
print("Using seed {}".format(rng_seed))
rng = random.Random(rng_seed)
print(os.environ.get('LEAF_DATA_META_DIR'))
if os.environ.get('LEAF_DATA_META_DIR') is not None:
    seed_fname = os.path.join(os.environ.get('LEAF_DATA_META_DIR'), SEED_FILES['sampling'])
    with open(seed_fname, 'w+') as f:
        f.write("# sampling_seed used by sampling script - supply as "
                "--smplseed to preprocess.sh or --seed to utils/sample.py\n")
        f.write(str(rng_seed))
    print("- random seed written out to {file}".format(file=seed_fname))
else:
    print("- using random seed '{seed}' for sampling".format(seed=rng_seed))


def iid_sample(all_x_samples, all_y_samples, samples_to_take):
    indices = np.random.choice(np.arange(len(all_x_samples)), samples_to_take)
    sampled_x, sampled_y = [all_x_samples[i] for i in indices.tolist()], [all_y_samples[i] for i in indices.tolist()]
    return sampled_x, sampled_y


def niid_sample(all_x_samples, all_y_samples, samples_to_take, labels_distributions):
    p_dist = [labels_distributions[label] for label in all_y_samples]
    indices = np.random.choice(np.arange(len(all_x_samples)), samples_to_take, p_dist)
    sampled_x, sampled_y = [all_x_samples[i] for i in indices.tolist()], [all_y_samples[i] for i in indices.tolist()]
    return sampled_x, sampled_y

# Read options from args
n_workers = args.u
labels_dist = args.sampling_mode.split('+')[0]
assert labels_dist in ['iid', 'niid']
n_samples_dist = args.sampling_mode.split('+')[1]
assert n_samples_dist in ['sim', 'nsim']

tot_n_samples = 0
list_of_classes = []
for f in files:
    file_dir = os.path.join(subdir, f)
    with open(file_dir, 'r') as inf:
        data = json.load(inf, object_pairs_hook=OrderedDict)
    tot_n_samples += sum(data['num_samples'])
    raw_list = list(data['user_data'].values())
    raw_y = [elem['y'] for elem in raw_list]
    y_list = [item for sublist in raw_y for item in sublist]  # flatten raw_y
    list_of_classes += list(set(y_list))
    list_of_classes = list(set(list_of_classes))
n_classes = len(list_of_classes)
if n_samples_dist == 'sim':
    avg_samples_per_worker = tot_n_samples / n_workers
    if avg_samples_per_worker > args.spw:
        avg_samples_per_worker = args.spw
        samples_per_worker = (np.minimum(avg_samples_per_worker * (1 + np.random.normal(0, 0.1, n_workers)),
                                         np.asarray([args.spw for _ in range(n_workers)]))).tolist()
    else:
        samples_per_worker = (avg_samples_per_worker*(1+np.random.normal(0, 0.1, n_workers))).tolist()
else:
    avg_samples_per_worker = tot_n_samples / n_workers
    if avg_samples_per_worker > args.spw:
        size = args.spw * 2
    else:
        size = avg_samples_per_worker * 2
    samples_per_worker = np.random.uniform(0.1*size, size, n_workers).tolist()
samples_per_worker = [round(sample) for sample in samples_per_worker]
print('samples_per_worker: {}'.format(samples_per_worker))

workers = [str(i) for i in range(n_workers)]
if labels_dist == 'niid':
    workers_labels_distributions = {w: softmax(np.random.random_integers(0, 100, n_classes)) for w in workers}
else:
    workers_labels_distributions = {w: 1./n_workers for w in workers}
all_data = {}
all_data['users'] = workers
all_data['num_samples'] = [0 for i in range(n_workers)]
all_data['user_data'] = {worker: {'x': [], 'y': []} for worker in workers}

for f in files:
    print('processing file {}...'.format(f))
    file_dir = os.path.join(subdir, f)
    with open(file_dir, 'r') as inf:
        data = json.load(inf, object_pairs_hook=OrderedDict)
    n_samples = sum(data['num_samples'])
    num_new_samples = int(args.fraction * n_samples)
    indices = [i for i in range(n_samples)]
    new_indices = rng.sample(indices, num_new_samples)
    sample_percentage = n_samples/tot_n_samples
    for i, worker in enumerate(workers):
        samples_to_take = round(sample_percentage*samples_per_worker[i])
        raw_list = list(data['user_data'].values())
        raw_x = [elem['x'] for elem in raw_list]
        raw_y = [elem['y'] for elem in raw_list]
        x_list = [item for sublist in raw_x for item in sublist]  # flatten raw_x
        y_list = [item for sublist in raw_y for item in sublist]  # flatten raw_y
        all_x_samples = [x_list[i] for i in new_indices]
        all_y_samples = [y_list[i] for i in new_indices]
        if labels_dist == 'iid':
            sampled_x, sampled_y = iid_sample(all_x_samples, all_y_samples, samples_to_take)
        else:
            sampled_x, sampled_y = niid_sample(all_x_samples, all_y_samples,
                                               samples_to_take, workers_labels_distributions[worker])

        all_data['num_samples'][i] += samples_to_take
        all_data['user_data'][worker]['x'] += sampled_x
        all_data['user_data'][worker]['y'] += sampled_y

slabel = args.sampling_mode

arg_frac = str(args.fraction)
arg_frac = arg_frac[2:]
arg_nu = str(args.u)
arg_nu = arg_nu[2:]
arg_label = arg_frac
if (args.iid):
    arg_label = '%s_%s' % (arg_nu, arg_label)
file_name = '%s_%s_%s.json' % ((f[:-5]), slabel, arg_label)
ouf_dir = os.path.join(data_dir, '{}_workers'.format(n_workers), 'spw={}'.format(args.spw), 'mode={}'.format(args.sampling_mode), 'sampled_data', file_name)

print('writing %s' % file_name)
with open(ouf_dir, 'w') as outfile:
    json.dump(all_data, outfile)


