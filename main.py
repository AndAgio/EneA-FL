import argparse
import numpy as np
import random
import math
import torch
import os
import json
from base64 import b64encode

from enea_fl.federation import Federation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name of dataset;', type=str, choices=['sent140', 'femnist', 'mnist'],
                        required=True)
    parser.add_argument('--num_workers', help='number of rounds to simulate;', type=int, default=100)
    parser.add_argument('--max_spw', help='maximum number of samples for each worker;', type=int, default=math.inf)
    parser.add_argument('--sampling_mode', help='mode to federate dataset; '
                                                'options are: [iid+sim, iid_nsim, niid+sim, niid_nsim]',
                        type=str, default='iid+sim')
    parser.add_argument('--clients_per_round', help='number of clients trained per round;', type=int, default=20)
    parser.add_argument('--batch_size', help='batch size when clients train on data;', type=int, default=10)
    parser.add_argument('--seed', help='seed for random client sampling and batch splitting', type=int, default=1234)
    parser.add_argument('--metrics_name', help='name for metrics file;', type=str, default='metrics', required=False)
    parser.add_argument('--use_val_set', help='use validation set;', action='store_true')
    parser.add_argument('--lr', help='learning rate for local optimizers;', type=float, default=-1, required=False)

    parser.add_argument('--policy', help='workers selection policy;', type=str,
                        choices=['random', 'energy_aware', 'acc_aware', 'oort', 'oort_v2'],
                        required=False, default='energy_aware')
    parser.add_argument('--alpha', help='alpha parameter;', type=float, default=0.5, required=False)
    parser.add_argument('--beta', help='beta parameter;', type=float, default=0.5, required=False)
    parser.add_argument('--k', help='k parameter;', type=float, default=0.9, required=False)

    parser.add_argument('--raspberry_p', help='percentage of raspberry pi devices;',
                        type=float, default=1./7, required=False)
    parser.add_argument('--nano_cpu_p', help='percentage of jetson nano with only cpu devices;',
                        type=float, default=1./7, required=False)
    parser.add_argument('--nano_gpu_p', help='percentage of jetson nano with gpu devices;',
                        type=float, default=1./7, required=False)
    parser.add_argument('--xavier_cpu_p', help='percentage of jetson nano with only cpu devices;',
                        type=float, default=1./7, required=False)
    parser.add_argument('--xavier_gpu_p', help='percentage of jetson nano with gpu devices;',
                        type=float, default=1./7, required=False)
    parser.add_argument('--orin_cpu_p', help='percentage of jetson orin with only cpu devices;',
                        type=float, default=1./7, required=False)
    parser.add_argument('--orin_gpu_p', help='percentage of jetson orin with gpu devices;',
                        type=float, default=1./7, required=False)

    parser.add_argument('--random_death', help='randomly kill workers after some time;', action='store_true')

    parser.add_argument('--max_update_latency', help='maximum amount of time that the server waits for updates;',
                        type=float, default=None, required=False)

    parser.add_argument('--target_type', help='type of target to reach during training; '
                                              'options are: [rounds, acc, energy, time]',
                        type=str, default='rounds')
    parser.add_argument('--target_value', help='value of target to reach during training; ',
                        type=float, default=100)

    parser.add_argument('--cuda', help='type of cuda device to use in simulation. '
                                       'All federation will fit in a single cuda device; '
                                       'options are: [cpu, 0, 1]',
                        type=str, default='cpu')

    return parser.parse_args()


def store_sim_id_params(sim_id, args):
    metrics_dir = os.path.join('metrics', sim_id)
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    logs_dir = os.path.join('logs', sim_id)
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print('Simulation with ID {} has arguments:\n\t{}'.format(sim_id,
                                                              args.__dict__))


def define_device_type_distribution(args):
    distribution = {'raspberrypi': args.raspberry_p,
                    'nano_cpu': args.nano_cpu_p,
                    'nano_gpu': args.nano_gpu_p,
                    'xavier_cpu': args.xavier_cpu_p,
                    'xavier_gpu': args.xavier_gpu_p,
                    'orin_cpu': args.orin_cpu_p,
                    'orin_gpu': args.orin_gpu_p}
    # assert sum(list(distribution.values())) == 1
    return distribution


def main():
    args = parse_args()
    # Set the random seed if provided (affects client sampling, and batching)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    sim_id = '######'
    while not sim_id.isalnum():
        random_bytes = os.urandom(64)
        sim_id = b64encode(random_bytes).decode('utf-8')[:6]
    store_sim_id_params(sim_id, args)

    cuda_device = torch.device('cuda:{}'.format(args.cuda)
                               if torch.cuda.is_available() and args.cuda != 'cpu' else 'cpu')

    my_federation = Federation(dataset=args.dataset,
                               n_workers=args.num_workers,
                               device_types_distribution=define_device_type_distribution(args),
                               sampling_mode=args.sampling_mode,
                               max_spw=args.max_spw,
                               target_type=args.target_type,
                               target_value=args.target_value,
                               use_val_set=args.use_val_set,
                               random_workers_death=args.random_death,
                               sim_id=sim_id,
                               cuda_device=cuda_device)
    my_federation.run(clients_per_round=args.clients_per_round,
                      batch_size=args.batch_size,
                      lr=args.lr,
                      policy=args.policy,
                      alpha=args.alpha,
                      beta=args.beta,
                      k=args.k,
                      max_update_latency=args.max_update_latency)


if __name__ == '__main__':
    main()
