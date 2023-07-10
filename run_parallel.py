import os
import subprocess
import argparse
import numpy as np


def run_all_in_parallel(commands, files):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    procs = [subprocess.Popen(commands[i], shell=True, stdout=open(files[i], "w")) for i in range(len(commands))]
    for p in procs:
        p.wait()


def run_alpha_beta():
    datasets = ["femnist", "sent140"]
    alphas = [i for i in np.arange(0, 1.05, 0.05)]
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha={} --beta={} "
                "--target_type='rounds' --target_value=100".format(d, alphas[i], 1 - alphas[i])
                for d in datasets for i in range(len(alphas))]
    commands += ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                 "--clients_per_round=20 --lr=0.1 --policy='random' --target_type='rounds' --target_value=100".format(d)
                 for d in datasets]
    os.makedirs(os.path.join('logs', 'alphas'), exist_ok=True)
    logfiles = ["logs/alphas/d={}-a={}-b={}.txt".format(d, alphas[i], 1 - alphas[i])
                for d in datasets
                for i in range(len(alphas))]
    logfiles += ["logs/alphas/d={}-random.txt".format(d) for d in datasets]
    run_all_in_parallel(commands, logfiles)


def run_clients():
    datasets = ["femnist", "sent140"]
    clients = [i for i in range(10, 80, 5)]
    policies = ['random', 'energy_aware']
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round={} --lr=0.1 --policy={} --alpha=0.7 --beta=0.3"
                "--target_type='rounds' --target_value=100".format(d, client, policy)
                for d in datasets
                for client in clients
                for policy in policies]
    os.makedirs(os.path.join('logs', 'clients'), exist_ok=True)
    logfiles = ["logs/clients/d={}-c={}-p={}.txt".format(d, client, policy)
                for d in datasets
                for client in clients
                for policy in policies]
    run_all_in_parallel(commands, logfiles)


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
