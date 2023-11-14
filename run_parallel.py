import os
import random
import subprocess
import argparse
import numpy as np
import math
import time

BEST_ALPHA = 0.9
BEST_BETA = 1 - BEST_ALPHA
BEST_K = 0.9


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
    batch = 4.
    start_time = time.time()
    tot_batches = math.ceil(len(commands) / batch)
    for i in range(tot_batches):
        print('Running batch {}/{}. Time taken: {:.3f} s'.format(i, tot_batches, time.time() - start_time),
              end='\r')
        coms = commands[int(batch * i):int(batch * (i + 1))]
        logs = logfiles[int(batch * i):int(batch * (i + 1))]
        for cuda_device in range(0, 2):
            coms = [com + ' --cuda={}'.format('cpu' if 'sent140' in com else 0 if 0 <= i < batch / 2 else 1)
                    for i, com in enumerate(coms)]
        run_all_in_parallel(coms, logs)


def run_alpha_beta():
    datasets = ["mnist"]  # , "sent140"]
    n_experiments_for_setup = 10
    seeds = [random.randint(1000, 10000) for _ in range(n_experiments_for_setup)]
    os.makedirs(os.path.join('logs', 'alphas'), exist_ok=True)
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='random' --target_type='rounds' --target_value=30  "
                "--batch_size=10".format(d) for d in datasets for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/alphas/d={}-random-({}).txt".format(d, i) for d in datasets for i in
                range(n_experiments_for_setup)]
    print('Running all experiments in parallel for random policy.')
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    for dataset in datasets:
        print('Running all experiments in parallel for dataset: {}'.format(dataset))
        alphas = [i for i in np.arange(0, 1.05, 0.2)]
        betas = [j for j in np.arange(0, 100.05, 20.)]
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                    "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha={} --beta={} --k=0.9"
                    " --target_type='rounds' --target_value=30  --batch_size=10".format(dataset,
                                                                                        alphas[i],
                                                                                        betas[j])
                    for i in range(len(alphas))
                    for j in range(len(betas))
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/alphas/d={}-a={}-b={}-({}).txt".format(dataset, alphas[i], betas[j], k)
                    for i in range(len(alphas))
                    for j in range(len(betas))
                    for k in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)


def run_k():
    datasets = ["mnist"]  # , "sent140"]
    n_experiments_for_setup = 10
    os.makedirs(os.path.join('logs', 'ks'), exist_ok=True)
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='random' --target_type='rounds' --target_value=30  "
                "--batch_size=10".format(d) for d in datasets for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/ks/d={}-random-({}).txt".format(d, i) for d in datasets for i in
                range(n_experiments_for_setup)]
    print('Running all experiments in parallel for random policy over the two datasets')
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    for dataset in datasets:
        print('Running all experiments in parallel for dataset: {}'.format(dataset))
        ks = [i for i in np.arange(0, 1.05, 0.1)]
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                    "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k={}"
                    " --target_type='rounds' --target_value=30  --batch_size=10".format(dataset,
                                                                                        ks[i])
                    for i in range(len(ks))
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/ks/d={}-k={}-({}).txt".format(dataset, ks[i], j)
                    for i in range(len(ks))
                    for j in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)


def run_dev_ps():
    datasets = ["mnist"]  # , "sent140"]
    n_experiments_for_setup = 10
    os.makedirs(os.path.join('logs', 'probs'), exist_ok=True)
    for dataset in datasets:
        print('Running all experiments in parallel for dataset and random sampling: {}'.format(dataset))
        probs = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.25],
                 [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7],
                 [0.9/5, 0.9/5, 0.9/5, 0.9/5, 0.9/5, 0.05, 0.05], ]
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                    "--clients_per_round=20 --lr=0.1 --policy='random'"
                    " --target_type='acc' --target_value=0.97  --batch_size=10"
                    " --raspberry_p={} --orin_gpu_p={} --orin_cpu_p={} "
                    "--nano_cpu_p={} --xavier_cpu_p={} --xavier_gpu_p={} "
                    "--nano_gpu_p={}".format(dataset, probs[i][0], probs[i][1], probs[i][2], probs[i][3],
                                             probs[i][4], probs[i][5], probs[i][6])
                    for i in range(len(probs))
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/probs/random-d={}-p={}-({}).txt".format(dataset, probs[i], j)
                    for i in range(len(probs))
                    for j in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)

    for dataset in datasets:
        print('Running all experiments in parallel for dataset: {}'.format(dataset))
        probs = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.25, 0.25],
                 [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7],
                 [0.9/5, 0.9/5, 0.9/5, 0.9/5, 0.9/5, 0.05, 0.05], ]
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                    "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k=0.8"
                    " --target_type='acc' --target_value=0.97  --batch_size=10"
                    " --raspberry_p={} --orin_gpu_p={} --orin_cpu_p={} "
                    "--nano_cpu_p={} --xavier_cpu_p={} --xavier_gpu_p={} "
                    "--nano_gpu_p={}".format(dataset, probs[i][0], probs[i][1], probs[i][2], probs[i][3],
                                             probs[i][4], probs[i][5], probs[i][6])
                    for i in range(len(probs))
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/probs/ene-d={}-p={}-({}).txt".format(dataset, probs[i], j)
                    for i in range(len(probs))
                    for j in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)


def run_nsim():
    datasets = ["mnist"]  # , "sent140"]
    n_experiments_for_setup = 10
    os.makedirs(os.path.join('logs', 'modes'), exist_ok=True)
    for dataset in datasets:
        print('Running all experiments in parallel for dataset and random sampling: {}'.format(dataset))
        modes = ['iid+nsim']
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode={} "
                    "--clients_per_round=20 --lr=0.1 --policy='random'"
                    " --target_type='rounds' --target_value=30  --batch_size=10".format(dataset,
                                                                                        modes[i])
                    for i in range(len(modes))
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/modes/random-d={}-m={}-({}).txt".format(dataset, modes[i], j)
                    for i in range(len(modes))
                    for j in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)

    for dataset in datasets:
        print('Running all experiments in parallel for dataset: {}'.format(dataset))
        modes = ['iid+nsim']
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode={} "
                    "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k=0.8"
                    " --target_type='rounds' --target_value=30  --batch_size=10".format(dataset,
                                                                                        modes[i])
                    for i in range(len(modes))
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/modes/ene-d={}-m={}-({}).txt".format(dataset, modes[i], j)
                    for i in range(len(modes))
                    for j in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)


def run_clients():
    datasets = ["mnist"]  # , "sent140"]
    n_experiments_for_setup = 10
    # os.makedirs(os.path.join('logs', 'clients', 'random'), exist_ok=True)
    # for dataset in datasets:
    #     print('Running all experiments in parallel for dataset and random sampling: {}'.format(dataset))
    #     clients = [i for i in range(10, 90, 10)]
    #     commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
    #                 "--clients_per_round={} --lr=0.1 --policy='random'"
    #                 " --target_type='acc' --target_value=0.97  --batch_size=10".format(dataset,
    #                                                                                    clients[i])
    #                 for i in range(len(clients))
    #                 for _ in range(n_experiments_for_setup)]
    #     logfiles = ["logs/clients/random/d={}-c={}-({}).txt".format(dataset, clients[i], j)
    #                 for i in range(len(clients))
    #                 for j in range(n_experiments_for_setup)]
    #     print('Number of experiments: {}.'.format(len(commands)))
    #     run_in_batch(commands, logfiles)

    # os.makedirs(os.path.join('logs', 'clients', 'enea'), exist_ok=True)
    # for dataset in datasets:
    #     print('Running all experiments in parallel for dataset: {}'.format(dataset))
    #     clients = [i for i in range(10, 90, 10)]
    #     commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
    #                 "--clients_per_round={} --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k=0.8"
    #                 " --target_type='acc' --target_value=0.97  --batch_size=10".format(dataset,
    #                                                                                    clients[i])
    #                 for i in range(len(clients))
    #                 for _ in range(n_experiments_for_setup)]
    #     logfiles = ["logs/clients/enea/d={}-c={}-({}).txt".format(dataset, clients[i], j)
    #                 for i in range(len(clients))
    #                 for j in range(n_experiments_for_setup)]
    #     print('Number of experiments: {}.'.format(len(commands)))
    #     run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'clients', 'oort'), exist_ok=True)
    for dataset in datasets:
        print('Running all experiments in parallel for dataset: {}'.format(dataset))
        clients = [i for i in range(10, 90, 10)]
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                    "--clients_per_round={} --lr=0.1 --policy='oort' --k=0.8"
                    " --target_type='acc' --target_value=0.97  --batch_size=10".format(dataset,
                                                                                       clients[i])
                    for i in range(len(clients))
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/clients/oort/d={}-c={}-({}).txt".format(dataset, clients[i], j)
                    for i in range(len(clients))
                    for j in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'clients', 'oort_v2'), exist_ok=True)
    for dataset in datasets:
        print('Running all experiments in parallel for dataset: {}'.format(dataset))
        clients = [i for i in range(10, 90, 10)]
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                    "--clients_per_round={} --lr=0.1 --policy='oort_v2' --k=0.8"
                    " --target_type='acc' --target_value=0.97  --batch_size=10".format(dataset,
                                                                                       clients[i])
                    for i in range(len(clients))
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/clients/oort_v2/d={}-c={}-({}).txt".format(dataset, clients[i], j)
                    for i in range(len(clients))
                    for j in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)


def run_death():
    datasets = ["mnist"]  # , "sent140"]
    n_experiments_for_setup = 10
    # os.makedirs(os.path.join('logs', 'deaths', 'random'), exist_ok=True)
    # for dataset in datasets:
    #     print('Running all experiments in parallel for dataset and random sampling: {}'.format(dataset))
    #     deaths = ['--random_death']
    #     commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
    #                 "--clients_per_round=20 --lr=0.1 --policy='random'"
    #                 " --target_type='acc' --target_value=0.97  --batch_size=10 {}".format(dataset,
    #                                                                                        deaths[i])
    #                 for i in range(len(deaths))
    #                 for _ in range(n_experiments_for_setup)]
    #     logfiles = ["logs/deaths/random/d={}-de={}-({}).txt".format(dataset, deaths[i], j)
    #                 for i in range(len(deaths))
    #                 for j in range(n_experiments_for_setup)]
    #     print('Number of experiments: {}.'.format(len(commands)))
    #     run_in_batch(commands, logfiles)

    # os.makedirs(os.path.join('logs', 'deaths', 'enea'), exist_ok=True)
    # for dataset in datasets:
    #     print('Running all experiments in parallel for dataset: {}'.format(dataset))
    #     deaths = ['--random_death']
    #     commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
    #                 "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k=0.8"
    #                 " --target_type='acc' --target_value=0.97  --batch_size=10 {}".format(dataset,
    #                                                                                        deaths[i])
    #                 for i in range(len(deaths))
    #                 for _ in range(n_experiments_for_setup)]
    #     logfiles = ["logs/deaths/enea/d={}-de={}-({}).txt".format(dataset, deaths[i], j)
    #                 for i in range(len(deaths))
    #                 for j in range(n_experiments_for_setup)]
    #     print('Number of experiments: {}.'.format(len(commands)))
    #     run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'deaths', 'oort'), exist_ok=True)
    for dataset in datasets:
        print('Running all experiments in parallel for dataset: {}'.format(dataset))
        deaths = ['--random_death']
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                    "--clients_per_round=20 --lr=0.1 --policy='oort' --alpha=0.6 --beta=40 --k=0.8"
                    " --target_type='acc' --target_value=0.97  --batch_size=10 {}".format(dataset,
                                                                                           deaths[i])
                    for i in range(len(deaths))
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/deaths/oort/d={}-de={}-({}).txt".format(dataset, deaths[i], j)
                    for i in range(len(deaths))
                    for j in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'deaths', 'oort_v2'), exist_ok=True)
    for dataset in datasets:
        print('Running all experiments in parallel for dataset: {}'.format(dataset))
        deaths = ['--random_death']
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                    "--clients_per_round=20 --lr=0.1 --policy='oort_v2' --alpha=0.6 --beta=40 --k=0.8"
                    " --target_type='acc' --target_value=0.97  --batch_size=10 {}".format(dataset,
                                                                                           deaths[i])
                    for i in range(len(deaths))
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/deaths/oort_v2/d={}-de={}-({}).txt".format(dataset, deaths[i], j)
                    for i in range(len(deaths))
                    for j in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)


def run_energy_budget_mnist():
    dataset = "mnist"
    n_experiments_for_setup = 10
    os.makedirs(os.path.join('logs', 'energy_budget_mnist', 'random'), exist_ok=True)
    print('Running all experiments in parallel for dataset and random sampling: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='random'"
                " --target_type='energy' --target_value=1000000000  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_mnist/random/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'energy_budget_mnist', 'enea'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='energy' --target_value=1000000000  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_mnist/enea/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'energy_budget_mnist', 'oort'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='energy' --target_value=1000000000  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_mnist/oort/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'energy_budget_mnist', 'oort_v2'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort_v2' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='energy' --target_value=1000000000  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_mnist/oort_v2/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)


def run_energy_budget_nbaiot():
    dataset = "nbaiot"
    n_experiments_for_setup = 10
    os.makedirs(os.path.join('logs', 'energy_budget_nbaiot', 'random'), exist_ok=True)
    print('Running all experiments in parallel for dataset and random sampling: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='random'"
                " --target_type='energy' --target_value=5000000  --batch_size=10 --orin_gpu_p=0 --orin_cpu_p=0 --raspberry_p=0.2 --nano_cpu_p=0.2 --xavier_cpu_p=0.2 --xavier_gpu_p=0.2 --nano_gpu_p=0.2".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_nbaiot/random/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'energy_budget_nbaiot', 'enea'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='energy' --target_value=5000000  --batch_size=10 --orin_gpu_p=0 --orin_cpu_p=0 --raspberry_p=0.2 --nano_cpu_p=0.2 --xavier_cpu_p=0.2 --xavier_gpu_p=0.2 --nano_gpu_p=0.2".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_nbaiot/enea/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'energy_budget_nbaiot', 'oort'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='energy' --target_value=5000000  --batch_size=10 --orin_gpu_p=0 --orin_cpu_p=0 --raspberry_p=0.2 --nano_cpu_p=0.2 --xavier_cpu_p=0.2 --xavier_gpu_p=0.2 --nano_gpu_p=0.2".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_nbaiot/oort/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'energy_budget_nbaiot', 'oort_v2'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort_v2' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='energy' --target_value=5000000  --batch_size=10 --orin_gpu_p=0 --orin_cpu_p=0 --raspberry_p=0.2 --nano_cpu_p=0.2 --xavier_cpu_p=0.2 --xavier_gpu_p=0.2 --nano_gpu_p=0.2".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_nbaiot/oort_v2/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)


def run_energy_budget_sent():
    dataset = "sent140"
    n_experiments_for_setup = 10
    os.makedirs(os.path.join('logs', 'energy_budget_sent140', 'random'), exist_ok=True)
    print('Running all experiments in parallel for dataset and random sampling: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='random'"
                " --target_type='energy' --target_value=500000000  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_sent140/random/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'energy_budget_sent140', 'enea'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='energy' --target_value=500000000  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_sent140/enea/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'energy_budget_sent140', 'oort'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='energy' --target_value=500000000  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_sent140/oort/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'energy_budget_sent140', 'oort_v2'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort_v2' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='energy' --target_value=500000000  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/energy_budget_sent140/oort_v2/d={}-ene-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)


def run_time_budget_mnist():
    dataset = "mnist"
    n_experiments_for_setup = 10
    os.makedirs(os.path.join('logs', 'time_budget_mnist', 'random'), exist_ok=True)
    print('Running all experiments in parallel for dataset and random sampling: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='random'"
                " --target_type='time' --target_value=28800  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_mnist/random/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'time_budget_mnist', 'enea'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='time' --target_value=28800  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_mnist/enea/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)
    
    os.makedirs(os.path.join('logs', 'time_budget_mnist', 'oort'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='time' --target_value=28800  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_mnist/oort/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'time_budget_mnist', 'oort_v2'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort_v2' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='time' --target_value=28800  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_mnist/oort_v2/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)


def run_time_budget_nbaiot():
    dataset = "nbaiot"
    n_experiments_for_setup = 10
    os.makedirs(os.path.join('logs', 'time_budget_nbaiot', 'random'), exist_ok=True)
    print('Running all experiments in parallel for dataset and random sampling: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='random'"
                " --target_type='time' --target_value=120  --batch_size=10 --orin_gpu_p=0 --orin_cpu_p=0 --raspberry_p=0.2 --nano_cpu_p=0.2 --xavier_cpu_p=0.2 --xavier_gpu_p=0.2 --nano_gpu_p=0.2".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_nbaiot/random/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'time_budget_nbaiot', 'enea'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='time' --target_value=120  --batch_size=10 --orin_gpu_p=0 --orin_cpu_p=0 --raspberry_p=0.2 --nano_cpu_p=0.2 --xavier_cpu_p=0.2 --xavier_gpu_p=0.2 --nano_gpu_p=0.2".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_nbaiot/enea/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)
    
    os.makedirs(os.path.join('logs', 'time_budget_nbaiot', 'oort'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='time' --target_value=120  --batch_size=10 --orin_gpu_p=0 --orin_cpu_p=0 --raspberry_p=0.2 --nano_cpu_p=0.2 --xavier_cpu_p=0.2 --xavier_gpu_p=0.2 --nano_gpu_p=0.2".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_nbaiot/oort/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)
    
    os.makedirs(os.path.join('logs', 'time_budget_nbaiot', 'oort_v2'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort_v2' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='time' --target_value=120  --batch_size=10 --orin_gpu_p=0 --orin_cpu_p=0 --raspberry_p=0.2 --nano_cpu_p=0.2 --xavier_cpu_p=0.2 --xavier_gpu_p=0.2 --nano_gpu_p=0.2".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_nbaiot/oort_v2/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)


def run_time_budget_sent():
    dataset = "sent140"
    n_experiments_for_setup = 10
    os.makedirs(os.path.join('logs', 'time_budget_sent', 'random'), exist_ok=True)
    print('Running all experiments in parallel for dataset and random sampling: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='random'"
                " --target_type='time' --target_value=14400  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_sent/random/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

    os.makedirs(os.path.join('logs', 'time_budget_sent', 'enea'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='time' --target_value=14400  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_sent/enea/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)
    
    os.makedirs(os.path.join('logs', 'time_budget_sent', 'oort'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='time' --target_value=14400  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_sent/oort/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)
    
    os.makedirs(os.path.join('logs', 'time_budget_sent', 'oort_v2'), exist_ok=True)
    print('Running all experiments in parallel for dataset: {}'.format(dataset))
    commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                "--clients_per_round=20 --lr=0.1 --policy='oort_v2' --alpha=0.6 --beta=40 --k=0.8"
                " --target_type='time' --target_value=14400  --batch_size=10".format(dataset)
                for _ in range(n_experiments_for_setup)]
    logfiles = ["logs/time_budget_sent/oort_v2/d={}-time-({}).txt".format(dataset, j)
                for j in range(n_experiments_for_setup)]
    print('Number of experiments: {}.'.format(len(commands)))
    run_in_batch(commands, logfiles)

def run_acc_target():
    datasets = ["sent140"]  # , "mnist"]
    n_experiments_for_setup = 10
    os.makedirs(os.path.join('logs', 'acc_target'), exist_ok=True)
    for dataset in datasets:
        print('Running all experiments in parallel for dataset and random sampling: {}'.format(dataset))
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                    "--clients_per_round=20 --lr=0.1 --policy='random'"
                    " --target_type='acc' --target_value=0.75  --batch_size=10".format(dataset)
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/acc_target/random-d={}-acc-({}).txt".format(dataset, j)
                    for j in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)

    for dataset in datasets:
        print('Running all experiments in parallel for dataset: {}'.format(dataset))
        commands = ["python main.py --dataset='{}' --num_workers=100 --max_spw=1000 --sampling_mode='iid+sim' "
                    "--clients_per_round=20 --lr=0.1 --policy='energy_aware' --alpha=0.6 --beta=40 --k=0.8"
                    " --target_type='acc' --target_value=0.75  --batch_size=10".format(dataset)
                    for _ in range(n_experiments_for_setup)]
        logfiles = ["logs/acc_target/ene-d={}-acc-({}).txt".format(dataset, j)
                    for j in range(n_experiments_for_setup)]
        print('Number of experiments: {}.'.format(len(commands)))
        run_in_batch(commands, logfiles)


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
    elif options.experiment.lower() in ['k']:
        run_k()
    elif options.experiment.lower() in ['data_dist']:
        run_nsim()
    elif options.experiment.lower() in ['client']:
        run_clients()
    elif options.experiment.lower() in ['death']:
        run_death()
    elif options.experiment.lower() in ['dev_p']:
        run_dev_ps()
    elif options.experiment.lower() in ['energy_budget_mnist']:
        run_energy_budget_mnist()
    elif options.experiment.lower() in ['energy_budget_sent']:
        run_energy_budget_sent()
    elif options.experiment.lower() in ['energy_budget_nbaiot']:
        run_energy_budget_nbaiot()
    elif options.experiment.lower() in ['time_budget_mnist']:
        run_time_budget_mnist()
    elif options.experiment.lower() in ['time_budget_sent']:
        run_time_budget_sent()
    elif options.experiment.lower() in ['time_budget_nbaiot']:
        run_time_budget_nbaiot()
    elif options.experiment.lower() in ['acc_target']:
        run_acc_target()
    else:
        raise ValueError('Option "{}" is not a valid option!'.format(options.experiment))
