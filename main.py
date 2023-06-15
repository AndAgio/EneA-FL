import argparse
import numpy as np
import random
import torch

from enea_fl.federation import Federation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='name of dataset;', type=str, choices=['sent140', 'femnist'], required=True)
    parser.add_argument('--num_workers', help='number of rounds to simulate;', type=int, default=100)
    parser.add_argument('--max_spw', help='maximum number of samples for each worker;', type=int, default=10000)
    parser.add_argument('--iid', help='true to federate dataset with iid', type=str, default='true')
    parser.add_argument('--num_rounds', help='number of rounds to simulate;', type=int, default=100)
    parser.add_argument('--eval_every', help='evaluate every ____ rounds;', type=int, default=1)
    parser.add_argument('--clients_per_round', help='number of clients trained per round;', type=int, default=20)
    parser.add_argument('--batch_size', help='batch size when clients train on data;', type=int, default=10)
    parser.add_argument('--seed', help='seed for random client sampling and batch splitting', type=int, default=0)
    parser.add_argument('--metrics_name', help='name for metrics file;', type=str, default='metrics', required=False)
    parser.add_argument('--use_val_set', help='use validation set;', action='store_true')
    parser.add_argument('--lr', help='learning rate for local optimizers;', type=float, default=-1, required=False)
    return parser.parse_args()


def main():
    args = parse_args()
    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    my_federation = Federation(dataset=args.dataset,
                               n_workers=args.num_workers,
                               max_spw=args.max_spw,
                               iid=True if args.iid.lower() in ['true', 't'] else False,
                               n_rounds=args.num_rounds,
                               use_val_set=args.use_val_set)
    my_federation.run(clients_per_round=args.clients_per_round,
                      batch_size=args.batch_size,
                      eval_every=args.eval_every)


if __name__ == '__main__':
    main()
