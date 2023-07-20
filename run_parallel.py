import os
import random
import subprocess
import argparse
import numpy as np
import math


def run_all_in_parallel(commands, files):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    procs = [subprocess.Popen(commands[i], shell=True, stdout=open(files[i], "w")) for i in range(len(commands))]
    for p in procs:
        p.wait()


def run_all_sequential(commands, files):
    os.makedirs('logs', exist_ok=True)
    procs = [subprocess.check_call(commands[i], shell=True, stdout=open(files[i], "w")) for i in range(len(commands))]


def run_in_batch(commands, logfiles):
    batch = 8.
    for i in range(math.ceil(len(commands) / batch)):
        coms = commands[int(batch * i):int(batch * (i + 1))]
        logs = logfiles[int(batch * i):int(batch * (i + 1))]
        for cuda_device in range(0, 2):
            coms = [com + ' --cuda={}'.format(0 if 0 <= i < batch / 2 else 1)
                    for i, com in enumerate(coms)]
        run_all_in_parallel(coms, logs)


def run_alpha_beta():
    datasets = ["mnist"]  # , "sent140"]
    n_experiments_for_setup = 10
    seeds = [random.randint(1000, 10000) for _ in range(n_experiments_for_setup)]
    os.makedirs(os.path.join('logs', 'alphas'), exist_ok=True)
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='random' --target_type='rounds' --target_value=30  "
                "--batch_size=10".format(d) for d in datasets for _ in range(5)]
    logfiles = ["logs/alphas/d={}-random-({}).txt".format(d, i) for d in datasets for i in
                range(n_experiments_for_setup)]
    print('Running all experiments in parallel for random policy over the two datasets')
    run_in_batch(commands, logfiles)

    for dataset in datasets:
        print('Running all experiments in parallel for dataset: {}'.format(dataset))
        alphas = [i for i in np.arange(0, 1.05, 0.1)]
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                    "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha={} --beta={} --k=0.9"
                    " --target_type='rounds' --target_value=30  --batch_size=10".format(dataset,
                                                                                        alphas[i],
                                                                                        1 - alphas[i])
                    for i in range(len(alphas))
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/alphas/d={}-a={}-b={}-({}).txt".format(dataset, alphas[i], 1 - alphas[i], j)
                    for i in range(len(alphas))
                    for j in range(n_experiments_for_setup)]
        run_in_batch(commands, logfiles)


def run_clients():
    datasets = ["mnist", "sent140"]
    policies = ['random', 'energy_aware']
    os.makedirs(os.path.join('logs', 'clients'), exist_ok=True)
    for dataset in datasets:
        for policy in policies:
            print('Running all experiments in parallel for combination ({}, {})'.format(dataset, policy))
            clients = [i for i in range(10, 85, 5)]
            commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                        "--clients_per_round={} --lr=0.1 --policy={} --alpha=0.7 --beta=0.3"
                        "--target_type='rounds' --target_value=100 --batch_size=512".format(dataset, client, policy)
                        for client in clients]
            logfiles = ["logs/clients/d={}-c={}-p={}.txt".format(dataset, client, policy)
                        for client in clients]
            run_in_batch(commands, logfiles)
    # run_all_sequential(commands, logfiles)


# def run_k():
#     datas = ["b", "c", "s"]
#     degradations = ["l"]  # ["d", "n", "l", "m"]
#     predictors = ["u", "kins", "kill", "kbann"]
#     commands = ["python main.py run_experiments -d {} -t {} -p {}".format(d, e, p) for d in datas
#                 for e in degradations
#                 for p in predictors]
#     logfiles = ["logs/d={}-e={}-p={}.txt".format(d, e, p) for d in datas for e in degradations for p in predictors]
#     run_all_in_parallel(commands, logfiles)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run all in parallel options",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--experiment", type=str, default='alpha_beta',
                        help="alpha_beta to run all alpha_beta options, "
                             "client to run experiments concerning impact of number of clients, "
                             "k to run all k options, ")
    options = parser.parse_args()

    if options.experiment.lower() in ['alpha_beta']:
        run_alpha_beta()
    elif options.experiment.lower() in ['client']:
        run_clients()
    else:
        raise ValueError('Option "{}" is not a valid option!'.format(options.experiment))
