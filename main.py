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
    parser.add_argument('--dataset', help='name of dataset;', type=str, choices=['sent140', 'femnist'], required=True)
    parser.add_argument('--num_workers', help='number of rounds to simulate;', type=int, default=100)
    parser.add_argument('--max_spw', help='maximum number of samples for each worker;', type=int, default=math.inf)
    parser.add_argument('--sampling_mode', help='mode to federate dataset; '
                                                'options are: [iid+sim, iid_nsim, niid+sim, niid_nsim]',
                        type=str, default='iid+sim')
    parser.add_argument('--num_rounds', help='number of rounds to simulate;', type=int, default=100)
    parser.add_argument('--eval_every', help='evaluate every ____ rounds;', type=int, default=1)
    parser.add_argument('--clients_per_round', help='number of clients trained per round;', type=int, default=20)
    parser.add_argument('--batch_size', help='batch size when clients train on data;', type=int, default=10)
    parser.add_argument('--seed', help='seed for random client sampling and batch splitting', type=int, default=0)
    parser.add_argument('--metrics_name', help='name for metrics file;', type=str, default='metrics', required=False)
    parser.add_argument('--use_val_set', help='use validation set;', action='store_true')
    parser.add_argument('--lr', help='learning rate for local optimizers;', type=float, default=-1, required=False)

    parser.add_argument('--policy', help='workers selection policy;', type=str, choices=['random', 'energy_aware'],
                        required=False, default='energy_aware')
    parser.add_argument('--alpha', help='alpha parameter;', type=float, default=0.5, required=False)
    parser.add_argument('--beta', help='beta parameter;', type=float, default=0.5, required=False)
    parser.add_argument('--k', help='k parameter;', type=float, default=0.9, required=False)

    parser.add_argument('--raspberry_p', help='percentage of raspberry pi devices;',
                        type=float, default=0.2, required=False)
    parser.add_argument('--nano_cpu_p', help='percentage of jetson nano with only cpu devices;',
                        type=float, default=0.2, required=False)
    parser.add_argument('--nano_gpu_p', help='percentage of jetson nano with gpu devices;',
                        type=float, default=0.2, required=False)
    parser.add_argument('--orin_cpu_p', help='percentage of jetson orin with only cpu devices;',
                        type=float, default=0.2, required=False)
    parser.add_argument('--orin_gpu_p', help='percentage of jetson orin with gpu devices;',
                        type=float, default=0.2, required=False)

    return parser.parse_args()


def store_sim_id_params(sim_id, args):
    metrics_dir = os.path.join('metrics', sim_id)
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def define_device_type_distribution(args):
    distribution = {'raspberrypi': args.raspberry_p,
                    'nano_cpu': args.nano_cpu_p,
                    'nano_gpu': args.nano_gpu_p,
                    'orin_cpu': args.orin_cpu_p,
                    'orin_gpu': args.orin_gpu_p}
    assert sum(list(distribution.values())) == 1
    return distribution


def main():
    args = parse_args()
    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    random_bytes = os.urandom(64)
    sim_id = b64encode(random_bytes).decode('utf-8')[:6]
    store_sim_id_params(sim_id, args)

    my_federation = Federation(dataset=args.dataset,
                               n_workers=args.num_workers,
                               device_types_distribution=define_device_type_distribution(args),
                               sampling_mode=args.sampling_mode,
                               max_spw=args.max_spw,
                               n_rounds=args.num_rounds,
                               use_val_set=args.use_val_set)
    my_federation.run(clients_per_round=args.clients_per_round,
                      batch_size=args.batch_size,
                      lr=args.lr,
                      eval_every=args.eval_every,
                      policy=args.policy,
                      alpha=args.alpha,
                      beta=args.beta,
                      k=args.k,
                      sim_id=sim_id)


if __name__ == '__main__':
    main()
