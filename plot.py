import os
import numpy as np
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from matplotlib import rcParams

for font in font_manager.findSystemFonts("."):
    font_manager.fontManager.addfont(font)
rcParams['font.family'] = 'Palatino'
import pandas as pd
import json

BASE_DIR = "metrics_server/"
EXPERIMENTS = ['time_budget']  # ['energy_budget']  # ['clients']  # ['k']  # ['alpha_beta']


def plot_experiment(experiment):
    if experiment == 'alpha_beta':
        plot_alpha_beta()
    elif experiment == 'k':
        plot_k()
    elif experiment == 'clients':
        plot_clients()
    elif experiment == 'energy_budget':
        plot_energy_budget()
    elif experiment == 'time_budget':
        plot_time_budget()


def plot_alpha_beta():
    root_dir = os.path.join(BASE_DIR, 'alphas_beta_all_rounds_target_10_runs_each', 'metrics')
    sub_dirs = [x[0] for x in os.walk(root_dir)][1:]
    sim_ids = [dire.split('/')[-1] for dire in sub_dirs]
    print(sim_ids)
    policies = []
    alphas = []
    betas = []
    for sim_id in sim_ids:
        args_file = os.path.join(BASE_DIR, 'alphas_beta_all_rounds_target_10_runs_each', 'metrics', sim_id, 'args.txt')
        with open(args_file) as f:
            sim_arguments = f.read()
            sim_arguments = json.loads(sim_arguments)
            print(sim_arguments)
            policies.append(sim_arguments['policy'])
            if sim_arguments['policy'] == 'energy_aware':
                alphas.append(sim_arguments['alpha'])
                betas.append(sim_arguments['beta'])
    policies = list(set(policies))
    alphas = sorted(list(set(alphas)))
    betas = sorted(list(set(betas)))
    print(policies)
    print(alphas)
    print(betas)
    perf_dictionary = {'a{:.1f}_b{:.1f}'.format(alpha, beta): [] for alpha in alphas for beta in betas}
    if 'random' in policies:
        perf_dictionary['random'] = []
    for sim_id in sim_ids:
        args_file = os.path.join(BASE_DIR, 'alphas_beta_all_rounds_target_10_runs_each', 'metrics', sim_id, 'args.txt')
        with open(args_file) as f:
            sim_arguments = f.read()
            sim_arguments = json.loads(sim_arguments)
        policy = sim_arguments['policy']
        alpha = sim_arguments['alpha'] if policy == 'energy_aware' else None
        beta = sim_arguments['beta'] if policy == 'energy_aware' else None
        perf_file = os.path.join(BASE_DIR, 'alphas_beta_all_rounds_target_10_runs_each',
                                 'metrics', sim_id, 'final_metrics.csv')
        print(sim_id)
        perfs = pd.read_csv(perf_file)
        if policy == 'random':
            try:
                accs_list = list(perfs['accuracy'])
                first_acc_reached = list(filter(lambda i: i > 0.97, accs_list))[0]
                rounds = accs_list.index(first_acc_reached)
            except IndexError:
                rounds = None
            perf_dictionary['random'].append({'acc': list(perfs['accuracy'])[-1],
                                              'energy': list(perfs['tot_energy'])[-1],
                                              'time': list(perfs['tot_time'])[-1],
                                              'rounds': rounds})
        else:
            try:
                accs_list = list(perfs['accuracy'])
                first_acc_reached = list(filter(lambda i: i > 0.97, accs_list))[0]
                rounds = accs_list.index(first_acc_reached)
            except IndexError:
                rounds = None
            perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)].append({'acc': list(perfs['accuracy'])[-1],
                                                                           'energy': list(perfs['tot_energy'])[-1],
                                                                           'time': list(perfs['tot_time'])[-1],
                                                                           'rounds': rounds})
    print()
    xs = alphas
    ys = betas
    all_avg_energies = []
    for alpha in alphas:
        for beta in betas:
            all_avg_energies.append(
                np.mean([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)]]))
    all_avg_energies.append(np.mean([item['energy'] / 2000000 for item in perf_dictionary['random']]))
    min_ene = min(all_avg_energies)
    max_ene = max(all_avg_energies)
    import matplotlib
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "red"], N=1000)

    energies_avg = {'a': [],
                    'b': [],
                    'size': [],
                    'color': []}
    for alpha in alphas:
        for beta in betas:
            energies_avg['a'].append(alpha)
            energies_avg['b'].append(beta)
            ene = np.mean([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)]])
            energies_avg['size'].append(ene)
            energies_avg['color'].append((ene - min_ene) / (max_ene - min_ene))
    energies_std = {'a': [],
                    'b': [],
                    'size': [],
                    'color': []}
    for alpha in alphas:
        for beta in betas:
            energies_std['a'].append(alpha)
            energies_std['b'].append(beta)
            ene = np.mean([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)]])
            tot = ene + np.std(
                [item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)]])
            energies_std['size'].append(tot)
            energies_std['color'].append((ene - min_ene) / (max_ene - min_ene))
    print()

    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    colors = [i / 500 for i in np.arange(min_ene, max_ene, 100)]
    norm = Normalize(vmin=min_ene.item() / 500, vmax=max_ene.item() / 500)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(colors)

    plt.scatter(energies_avg['a'], energies_avg['b'],
                s=energies_avg['size'],
                c=cmap(energies_avg['color']),
                alpha=0.5, )
    # plt.scatter(energies_std['a'], energies_std['b'],
    #             s=energies_std['size'],
    #             c=cmap(energies_std['color']),
    #             alpha=0.25, )
    for i in range(len(energies_avg['a'])):
        plt.text(energies_avg['a'][i] - 0.05, energies_avg['b'][i] - 2,
                 '{:.1f}'.format(energies_avg['size'][i] / 500), fontsize=12)
    plt.xlabel(r"$\alpha$", size=16)
    plt.xlim(-0.2, 1.2)
    plt.ylabel(r"$\beta$", size=16)
    plt.ylim(-20, 120)
    plt.title("Total energy spent by the federation", size=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar = plt.colorbar(mappable)
    cbar.ax.set_ylabel('[MJ]', size=16)
    os.makedirs('plots/alpha_beta', exist_ok=True)
    plt.savefig('plots/alpha_beta/energy_alpha_vs_beta.pdf')
    plt.show()

    comparison = {'x': [],
                  'y': [],
                  'avg': [],
                  'std': [],
                  'color': []}

    random_ene_avg = np.mean([item['energy'] / 2000000 for item in perf_dictionary['random']])
    random_ene_std = np.std([item['energy'] / 2000000 for item in perf_dictionary['random']])
    comparison['x'].append(0)
    comparison['y'].append(0)
    comparison['avg'].append(random_ene_avg)
    comparison['std'].append(random_ene_std)
    comparison['color'].append((random_ene_avg - min_ene) / (max_ene - min_ene))

    worst_ene_avg = np.mean([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0., 100)]])
    worst_ene_std = np.std([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0., 100)]])
    comparison['x'].append(0.5)
    comparison['y'].append(0)
    comparison['avg'].append(worst_ene_avg)
    comparison['std'].append(worst_ene_std)
    comparison['color'].append((worst_ene_avg - min_ene) / (max_ene - min_ene))

    best_ene_avg = np.mean([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0.6, 40)]])
    best_ene_std = np.std([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0.6, 40)]])
    comparison['x'].append(1)
    comparison['y'].append(0)
    comparison['avg'].append(best_ene_avg)
    comparison['std'].append(best_ene_std)
    comparison['color'].append((best_ene_avg - min_ene) / (max_ene - min_ene))

    plt.scatter(comparison['x'], comparison['y'],
                s=comparison['avg'],
                c=cmap(comparison['color']),
                alpha=0.5, )
    # plt.scatter(comparison['x'], comparison['y'],
    #             s=comparison['std'],
    #             c=cmap(comparison['color']),
    #             alpha=0.25, )
    for i in range(len(comparison['x'])):
        plt.text(comparison['x'][i] - 0.05, comparison['y'][i] - 0.02,
                 '{:.1f}'.format(comparison['avg'][i] / 500), fontsize=12)
    plt.xlim(-0.3, 1.2)
    plt.ylim(-0.5, 0.5)
    plt.title("Total energy spent by the federation", size=18)
    cbar = plt.colorbar(mappable)
    cbar.ax.set_ylabel('[MJ]', size=16)

    labels = ['Random', 'Worst', 'Best']
    locations = [0, 0.5, 1]

    plt.xticks(ticks=locations, labels=labels, size=16)
    plt.yticks(ticks=[], labels=[])
    plt.savefig('plots/alpha_beta/energy_random_vs_ours.pdf')
    plt.show()

    #### TIMES #######

    all_avg_times = []
    for alpha in alphas:
        for beta in betas:
            all_avg_times.append(
                np.mean([item['time'] / 40 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)]]))
    all_avg_times.append(np.mean([item['time'] / 40 for item in perf_dictionary['random']]))
    min_time = min(all_avg_times)
    max_time = max(all_avg_times)
    import matplotlib
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "red"], N=1000)

    times_avg = {'a': [],
                 'b': [],
                 'size': [],
                 'color': []}
    for alpha in alphas:
        for beta in betas:
            times_avg['a'].append(alpha)
            times_avg['b'].append(beta)
            time = np.mean([item['time'] / 40 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)]])
            times_avg['size'].append(time)
            times_avg['color'].append((time - min_time) / (max_time - min_time))
    times_std = {'a': [],
                 'b': [],
                 'size': [],
                 'color': []}
    for alpha in alphas:
        for beta in betas:
            times_std['a'].append(alpha)
            times_std['b'].append(beta)
            time = np.mean([item['time'] / 40 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)]])
            tot = time + np.std(
                [item['time'] / 40 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)]])
            times_std['size'].append(tot)
            times_std['color'].append((time - min_time) / (max_ene - min_time))
    print()

    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    colors = [i / 90 for i in np.arange(min_time, max_time, 100)]
    norm = Normalize(vmin=min_time.item() / 90, vmax=max_time.item() / 90)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(colors)

    plt.scatter(times_avg['a'], times_avg['b'],
                s=times_avg['size'],
                c=cmap(times_avg['color']),
                alpha=0.5, )
    for i in range(len(times_avg['a'])):
        plt.text(times_avg['a'][i] - 0.05, times_avg['b'][i] - 2,
                 '{:.1f}'.format(times_avg['size'][i] / 90), fontsize=12)
    plt.xlabel(r"$\alpha$", size=16)
    plt.xlim(-0.2, 1.2)
    plt.ylabel(r"$\beta$", size=16)
    plt.ylim(-20, 120)
    plt.title("Total time spent by the federation", size=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar = plt.colorbar(mappable)
    cbar.ax.set_ylabel('[hours]', size=16)
    plt.savefig('plots/alpha_beta/time_alpha_vs_beta.pdf')
    plt.show()

    comparison = {'x': [],
                  'y': [],
                  'avg': [],
                  'std': [],
                  'color': []}

    random_time_avg = np.mean([item['time'] / 40 for item in perf_dictionary['random']])
    random_time_std = np.std([item['time'] / 40 for item in perf_dictionary['random']])
    comparison['x'].append(0)
    comparison['y'].append(0)
    comparison['avg'].append(random_time_avg)
    comparison['std'].append(random_time_std)
    comparison['color'].append((random_time_avg - min_time) / (max_time - min_time))

    worst_time_avg = np.mean([item['time'] / 40 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0., 100)]])
    worst_time_std = np.std([item['time'] / 40 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0., 100)]])
    comparison['x'].append(0.5)
    comparison['y'].append(0)
    comparison['avg'].append(worst_time_avg)
    comparison['std'].append(worst_time_std)
    comparison['color'].append((worst_time_avg - min_time) / (max_time - min_time))

    best_time_avg = np.mean([item['time'] / 40 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0.6, 40)]])
    best_time_std = np.std([item['time'] / 40 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0.6, 40)]])
    comparison['x'].append(1)
    comparison['y'].append(0)
    comparison['avg'].append(best_time_avg)
    comparison['std'].append(best_time_std)
    comparison['color'].append((best_time_avg - min_time) / (max_time - min_time))

    plt.scatter(comparison['x'], comparison['y'],
                s=comparison['avg'],
                c=cmap(comparison['color']),
                alpha=0.5, )
    # plt.scatter(comparison['x'], comparison['y'],
    #             s=comparison['std'],
    #             c=cmap(comparison['color']),
    #             alpha=0.25, )
    for i in range(len(comparison['x'])):
        plt.text(comparison['x'][i] - 0.05, comparison['y'][i] - 0.02,
                 '{:.1f}'.format(comparison['avg'][i] / 90), fontsize=12)
    plt.xlim(-0.3, 1.2)
    plt.ylim(-0.5, 0.5)
    plt.title("Total time spent by the federation", size=18)
    cbar = plt.colorbar(mappable)
    cbar.ax.set_ylabel('[hours]', size=16)

    labels = ['Random', 'Worst', 'Best']
    locations = [0, 0.5, 1]

    plt.xticks(ticks=locations, labels=labels, size=16)
    plt.yticks(ticks=[], labels=[])
    plt.savefig('plots/alpha_beta/time_random_vs_ours.pdf')
    plt.show()

    #### ACCURACY #######

    all_avg_rounds = []
    for alpha in alphas:
        for beta in betas:
            all_avg_rounds.append(
                np.mean([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)] if
                         item['rounds'] is not None]))
    all_avg_rounds.append(
        np.mean([item['rounds'] * 50 for item in perf_dictionary['random'] if item['rounds'] is not None]))
    min_rounds = min(all_avg_rounds)
    max_rounds = max(all_avg_rounds)
    import matplotlib
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green", "red"], N=1000)

    rounds_avg = {'a': [],
                  'b': [],
                  'size': [],
                  'color': []}
    for alpha in alphas:
        for beta in betas:
            rounds_avg['a'].append(alpha)
            rounds_avg['b'].append(beta)
            rounds = np.mean([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)] if
                              item['rounds'] is not None])
            rounds_avg['size'].append(rounds)
            rounds_avg['color'].append((rounds - min_rounds) / (max_rounds - min_rounds))

    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    colors = [i / 50 for i in np.arange(min_rounds, max_rounds, 100)]
    norm = Normalize(vmin=min_rounds.item() / 50, vmax=max_rounds.item() / 50)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(colors)

    plt.scatter(rounds_avg['a'], rounds_avg['b'],
                s=rounds_avg['size'],
                c=cmap(rounds_avg['color']),
                alpha=0.5, )
    for i in range(len(rounds_avg['a'])):
        plt.text(rounds_avg['a'][i] - 0.05, rounds_avg['b'][i] - 2,
                 '{:.1f}'.format(rounds_avg['size'][i] / 50), fontsize=12)
    plt.xlabel(r"$\alpha$", size=16)
    plt.xlim(-0.2, 1.2)
    plt.ylabel(r"$\beta$", size=16)
    plt.ylim(-20, 120)
    plt.title("Average rounds to reach 97% accuracy", size=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    cbar = plt.colorbar(mappable)
    cbar.ax.set_ylabel('rounds', size=16)
    plt.savefig('plots/alpha_beta/rounds_alpha_vs_beta.pdf')
    plt.show()

    comparison = {'x': [],
                  'y': [],
                  'avg': [],
                  'std': [],
                  'color': []}

    random_rounds_avg = np.mean(
        [item['rounds'] * 50 for item in perf_dictionary['random'] if item['rounds'] is not None])
    random_rounds_std = np.std(
        [item['rounds'] * 50 for item in perf_dictionary['random'] if item['rounds'] is not None])
    comparison['x'].append(0)
    comparison['y'].append(0)
    comparison['avg'].append(random_rounds_avg)
    comparison['std'].append(random_rounds_std)
    comparison['color'].append((random_rounds_avg - min_rounds) / (max_rounds - min_rounds))

    worst_rounds_avg = np.mean([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0., 100)] if
                                item['rounds'] is not None])
    worst_rounds_std = np.std([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0., 100)] if
                               item['rounds'] is not None])
    comparison['x'].append(0.5)
    comparison['y'].append(0)
    comparison['avg'].append(worst_rounds_avg)
    comparison['std'].append(worst_rounds_std)
    comparison['color'].append((worst_rounds_avg - min_rounds) / (max_rounds - min_rounds))

    best_rounds_avg = np.mean([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0.6, 40)] if
                               item['rounds'] is not None])
    best_rounds_std = np.std([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0.6, 40)] if
                              item['rounds'] is not None])
    comparison['x'].append(1)
    comparison['y'].append(0)
    comparison['avg'].append(best_rounds_avg)
    comparison['std'].append(best_rounds_std)
    comparison['color'].append((best_rounds_avg - min_rounds) / (max_rounds - min_rounds))

    plt.scatter(comparison['x'], comparison['y'],
                s=comparison['avg'],
                c=cmap(comparison['color']),
                alpha=0.5, )
    for i in range(len(comparison['x'])):
        plt.text(comparison['x'][i] - 0.05, comparison['y'][i] - 0.02,
                 '{:.1f}'.format(comparison['avg'][i] / 50), fontsize=12)
    plt.xlim(-0.3, 1.2)
    plt.ylim(-0.5, 0.5)
    plt.title("Average rounds to reach 97% accuracy", size=18)
    cbar = plt.colorbar(mappable)
    cbar.ax.set_ylabel('[rounds]', size=16)

    labels = ['Random', 'Worst', 'Best']
    locations = [0, 0.5, 1]

    plt.xticks(ticks=locations, labels=labels, size=16)
    plt.yticks(ticks=[], labels=[])
    plt.savefig('plots/alpha_beta/rounds_random_vs_ours.pdf')
    plt.show()


def plot_k():
    root_dir = os.path.join(BASE_DIR, 'ks_all_rounds_target_10_runs_each', 'metrics')
    sub_dirs = [x[0] for x in os.walk(root_dir)][1:]
    sim_ids = [dire.split('/')[-1] for dire in sub_dirs]
    print(sim_ids)
    policies = []
    ks = []
    for sim_id in sim_ids:
        args_file = os.path.join(BASE_DIR, 'ks_all_rounds_target_10_runs_each', 'metrics', sim_id, 'args.txt')
        with open(args_file) as f:
            sim_arguments = f.read()
            sim_arguments = json.loads(sim_arguments)
            print(sim_arguments)
            policies.append(sim_arguments['policy'])
            if sim_arguments['policy'] == 'energy_aware':
                ks.append(sim_arguments['k'])
    policies = list(set(policies))
    ks = sorted(list(set(ks)))
    print(policies)
    print(ks)
    perf_dictionary = {'k{:.1f}'.format(k): [] for k in ks}
    if 'random' in policies:
        perf_dictionary['random'] = []
    for sim_id in sim_ids:
        args_file = os.path.join(BASE_DIR, 'ks_all_rounds_target_10_runs_each', 'metrics', sim_id, 'args.txt')
        with open(args_file) as f:
            sim_arguments = f.read()
            sim_arguments = json.loads(sim_arguments)
        policy = sim_arguments['policy']
        k = sim_arguments['k'] if policy == 'energy_aware' else None
        perf_file = os.path.join(BASE_DIR, 'ks_all_rounds_target_10_runs_each',
                                 'metrics', sim_id, 'final_metrics.csv')
        print(sim_id)
        perfs = pd.read_csv(perf_file)
        if policy == 'random':
            try:
                accs_list = list(perfs['accuracy'])
                first_acc_reached = list(filter(lambda i: i > 0.97, accs_list))[0]
                rounds = accs_list.index(first_acc_reached)
            except IndexError:
                rounds = None
                continue
            # perf_dictionary['random'].append({'acc': list(perfs['accuracy'])[-1],
            #                                   'energy': list(perfs['tot_energy'])[-1],
            #                                   'time': list(perfs['tot_time'])[-1],
            #                                   'rounds': rounds})
            perf_dictionary['random'].append({'acc': list(perfs['accuracy'])[rounds],
                                              'energy': list(perfs['tot_energy'])[rounds],
                                              'time': list(perfs['tot_time'])[rounds],
                                              'rounds': rounds})
        else:
            try:
                accs_list = list(perfs['accuracy'])
                first_acc_reached = list(filter(lambda i: i > 0.97, accs_list))[0]
                rounds = accs_list.index(first_acc_reached)
            except IndexError:
                rounds = None
                continue
            # perf_dictionary['k{:.1f}'.format(k)].append({'acc': list(perfs['accuracy'])[-1],
            #                                              'energy': list(perfs['tot_energy'])[-1],
            #                                              'time': list(perfs['tot_time'])[-1],
            #                                              'rounds': rounds})
            perf_dictionary['k{:.1f}'.format(k)].append({'acc': list(perfs['accuracy'])[rounds],
                                                         'energy': list(perfs['tot_energy'])[rounds],
                                                         'time': list(perfs['tot_time'])[rounds],
                                                         'rounds': rounds})
    print()
    os.makedirs('plots/ks', exist_ok=True)

    xs = ks
    list_avg = [np.mean([item['energy'] / 1000000 for item in perf_dictionary['k{:.1f}'.format(k)]]) for k in ks]
    list_std = [np.std([item['energy'] / 1000000 for item in perf_dictionary['k{:.1f}'.format(k)]]) for k in ks]
    plt.plot(xs, list_avg, c='#1b9e77', marker='^', label="EneA-FL")
    plt.fill_between(xs,
                     [list_avg[i] - list_std[i] for i in range(len(list_avg))],
                     [list_avg[i] + list_std[i] for i in range(len(list_avg))],
                     alpha=0.5, edgecolor='#1b9e77', facecolor='#1b9e77')
    random_list_avg = [np.mean([item['energy'] / 1000000 for item in perf_dictionary['random']]) for _ in ks]
    random_list_std = [np.std([item['energy'] / 1000000 for item in perf_dictionary['random']]) for _ in ks]
    plt.plot(xs, random_list_avg, c='#d95f02', marker='s', label="Random")
    plt.fill_between(xs,
                     [random_list_avg[i] - random_list_std[i] for i in range(len(random_list_avg))],
                     [random_list_avg[i] + random_list_std[i] for i in range(len(random_list_avg))],
                     alpha=0.5, edgecolor='#d95f02', facecolor='#d95f02')
    plt.xlabel(r"$k$", size=14)
    plt.ylabel("Energy [KJ]", size=14)
    # plt.ylim(0, 6000)
    plt.ylim(0, 2500)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower left', prop={'size': 14})
    plt.title(r'Energy consumption vs $k$', size=16)
    plt.savefig('plots/ks/energy_random_vs_ours.pdf')
    plt.show()

    list_avg = [np.mean([item['time'] / 3600 for item in perf_dictionary['k{:.1f}'.format(k)]]) for k in ks]
    list_std = [np.std([item['time'] / 3600 for item in perf_dictionary['k{:.1f}'.format(k)]]) for k in ks]
    plt.plot(xs, list_avg, c='#1b9e77', marker='^', label="EneA-FL")
    plt.fill_between(xs,
                     [list_avg[i] - list_std[i] for i in range(len(list_avg))],
                     [list_avg[i] + list_std[i] for i in range(len(list_avg))],
                     alpha=0.5, edgecolor='#1b9e77', facecolor='#1b9e77')
    random_list_avg = [np.mean([item['time'] / 3600 for item in perf_dictionary['random']]) for _ in ks]
    random_list_std = [np.std([item['time'] / 3600 for item in perf_dictionary['random']]) for _ in ks]
    plt.plot(xs, random_list_avg, c='#d95f02', marker='s', label="Random")
    plt.fill_between(xs,
                     [random_list_avg[i] - random_list_std[i] for i in range(len(random_list_avg))],
                     [random_list_avg[i] + random_list_std[i] for i in range(len(random_list_avg))],
                     alpha=0.5, edgecolor='#d95f02', facecolor='#d95f02')
    plt.xlabel(r"$k$", size=14)
    plt.ylabel("Time [hours]", size=14)
    # plt.ylim(5, 35)
    plt.ylim(4, 16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower left', prop={'size': 14})
    plt.title(r'Time vs $k$', size=16)
    plt.savefig('plots/ks/time_random_vs_ours.pdf')
    plt.show()

    list_avg = [np.mean([item['rounds'] for item in perf_dictionary['k{:.1f}'.format(k)] if item['rounds'] is not None])
                for k in ks]
    list_std = [np.std([item['rounds'] for item in perf_dictionary['k{:.1f}'.format(k)] if item['rounds'] is not None])
                for k in ks]
    plt.plot(xs, list_avg, c='#1b9e77', marker='^', label="EneA-FL")
    plt.fill_between(xs,
                     [list_avg[i] - list_std[i] for i in range(len(list_avg))],
                     [list_avg[i] + list_std[i] for i in range(len(list_avg))],
                     alpha=0.5, edgecolor='#1b9e77', facecolor='#1b9e77')
    random_list_avg = [np.mean([item['rounds'] for item in perf_dictionary['random'] if item['rounds'] is not None]) for
                       _ in ks]
    random_list_std = [np.std([item['rounds'] for item in perf_dictionary['random'] if item['rounds'] is not None]) for
                       _ in ks]
    plt.plot(xs, random_list_avg, c='#d95f02', marker='s', label="Random")
    plt.fill_between(xs,
                     [random_list_avg[i] - random_list_std[i] for i in range(len(random_list_avg))],
                     [random_list_avg[i] + random_list_std[i] for i in range(len(random_list_avg))],
                     alpha=0.5, edgecolor='#d95f02', facecolor='#d95f02')
    plt.xlabel(r"$k$", size=14)
    plt.ylabel("Rounds", size=14)
    plt.ylim(6, 17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower left', prop={'size': 14})
    plt.title(r'Rounds to converge vs $k$', size=16)
    plt.savefig('plots/ks/rounds_random_vs_ours.pdf')
    plt.show()


def plot_clients():
    root_dir = os.path.join(BASE_DIR, 'clients_all_acc_target_10_runs_each', 'metrics')
    sub_dirs = [x[0] for x in os.walk(root_dir)][1:]
    sim_ids = [dire.split('/')[-1] for dire in sub_dirs]
    print(sim_ids)
    policies = []
    clients = []
    for sim_id in sim_ids:
        args_file = os.path.join(BASE_DIR, 'clients_all_acc_target_10_runs_each', 'metrics', sim_id, 'args.txt')
        with open(args_file) as f:
            sim_arguments = f.read()
            sim_arguments = json.loads(sim_arguments)
            print(sim_arguments)
            policies.append(sim_arguments['policy'])
            if sim_arguments['policy'] == 'energy_aware':
                clients.append(sim_arguments['clients_per_round'])
    policies = list(set(policies))
    clients = sorted(list(set(clients)))
    print(policies)
    print(clients)
    perf_dictionary = {'random': {'c{}'.format(client): [] for client in clients},
                       'eneafl': {'c{}'.format(client): [] for client in clients}}
    for sim_id in sim_ids:
        args_file = os.path.join(BASE_DIR, 'clients_all_acc_target_10_runs_each', 'metrics', sim_id, 'args.txt')
        with open(args_file) as f:
            sim_arguments = f.read()
            sim_arguments = json.loads(sim_arguments)
        policy = sim_arguments['policy']
        client = sim_arguments['clients_per_round']
        perf_file = os.path.join(BASE_DIR, 'clients_all_acc_target_10_runs_each',
                                 'metrics', sim_id, 'final_metrics.csv')
        print(sim_id)
        perfs = pd.read_csv(perf_file)
        if policy == 'random':
            accs_list = list(perfs['accuracy'])
            if len(accs_list) > 20:
                continue
            else:
                perf_dictionary['random']['c{}'.format(client)].append({'acc': list(perfs['accuracy'])[-1],
                                                                        'energy': list(perfs['tot_energy'])[-1],
                                                                        'time': list(perfs['tot_time'])[-1],
                                                                        'rounds': len(list(perfs['accuracy']))})
        else:
            accs_list = list(perfs['accuracy'])
            if len(accs_list) > 20:
                continue
            else:
                perf_dictionary['eneafl']['c{}'.format(client)].append({'acc': list(perfs['accuracy'])[-1],
                                                                        'energy': list(perfs['tot_energy'])[-1],
                                                                        'time': list(perfs['tot_time'])[-1],
                                                                        'rounds': len(list(perfs['accuracy']))})
    print()
    os.makedirs('plots/clients', exist_ok=True)

    xs = clients
    list_avg = [np.mean([item['energy'] / 1000000 for item in perf_dictionary['eneafl']['c{}'.format(client)]])
                for client in clients]
    list_std = [np.std([item['energy'] / 1000000 for item in perf_dictionary['eneafl']['c{}'.format(client)]])
                for client in clients]
    plt.plot(xs, list_avg, c='#1b9e77', marker='^', label="EneA-FL")
    plt.fill_between(xs,
                     [list_avg[i] - list_std[i] for i in range(len(list_avg))],
                     [list_avg[i] + list_std[i] for i in range(len(list_avg))],
                     alpha=0.5, edgecolor='#1b9e77', facecolor='#1b9e77')
    random_list_avg = [np.mean([item['energy'] / 1000000 for item in perf_dictionary['random']['c{}'.format(client)]])
                       for client in clients]
    random_list_std = [np.std([item['energy'] / 1000000 for item in perf_dictionary['random']['c{}'.format(client)]])
                       for client in clients]
    plt.plot(xs, random_list_avg, c='#d95f02', marker='s', label="Random")
    plt.fill_between(xs,
                     [random_list_avg[i] - random_list_std[i] for i in range(len(random_list_avg))],
                     [random_list_avg[i] + random_list_std[i] for i in range(len(random_list_avg))],
                     alpha=0.5, edgecolor='#d95f02', facecolor='#d95f02')
    plt.xlabel("Number of clients", size=14)
    plt.ylabel("Energy [KJ]", size=14)
    plt.ylim(0, 8500)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', prop={'size': 14})
    plt.title(r'Energy consumption vs number of clients per round', size=16)
    plt.savefig('plots/clients/energy_random_vs_ours.pdf')
    plt.show()

    list_avg = [np.mean([item['time'] / 3600 for item in perf_dictionary['eneafl']['c{}'.format(client)]])
                for client in clients]
    list_std = [np.std([item['time'] / 3600 for item in perf_dictionary['eneafl']['c{}'.format(client)]])
                for client in clients]
    plt.plot(xs, list_avg, c='#1b9e77', marker='^', label="EneA-FL")
    plt.fill_between(xs,
                     [list_avg[i] - list_std[i] for i in range(len(list_avg))],
                     [list_avg[i] + list_std[i] for i in range(len(list_avg))],
                     alpha=0.5, edgecolor='#1b9e77', facecolor='#1b9e77')
    random_list_avg = [np.mean([item['time'] / 3600 for item in perf_dictionary['random']['c{}'.format(client)]])
                       for client in clients]
    random_list_std = [np.std([item['time'] / 3600 for item in perf_dictionary['random']['c{}'.format(client)]])
                       for client in clients]
    plt.plot(xs, random_list_avg, c='#d95f02', marker='s', label="Random")
    plt.fill_between(xs,
                     [random_list_avg[i] - random_list_std[i] for i in range(len(random_list_avg))],
                     [random_list_avg[i] + random_list_std[i] for i in range(len(random_list_avg))],
                     alpha=0.5, edgecolor='#d95f02', facecolor='#d95f02')
    plt.xlabel("Number of clients", size=14)
    plt.ylabel("Time [hours]", size=14)
    # plt.ylim(5, 35)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', prop={'size': 14})
    plt.title(r'Time vs number of clients per round', size=16)
    plt.savefig('plots/clients/time_random_vs_ours.pdf')
    plt.show()


def plot_energy_budget():
    root_dir = os.path.join(BASE_DIR, 'ene_budget_all_acc_target_10_runs_each', 'metrics')
    sub_dirs = [x[0] for x in os.walk(root_dir)][1:]
    sim_ids = [dire.split('/')[-1] for dire in sub_dirs]
    print(sim_ids)
    policies = ['random', 'energy_aware']
    perf_dictionary = {pol: {'acc': [],
                             'energy': [],
                             'time': [],
                             'acc_list': []} for pol in policies}
    for sim_id in sim_ids:
        args_file = os.path.join(BASE_DIR, 'ene_budget_all_acc_target_10_runs_each', 'metrics', sim_id, 'args.txt')
        with open(args_file) as f:
            sim_arguments = f.read()
            sim_arguments = json.loads(sim_arguments)
        policy = sim_arguments['policy']
        perf_file = os.path.join(BASE_DIR, 'ene_budget_all_acc_target_10_runs_each',
                                 'metrics', sim_id, 'final_metrics.csv')
        print(sim_id)
        perfs = pd.read_csv(perf_file)
        accs_list = list(perfs['accuracy'])
        if accs_list[-2] < 0.9:
            continue
        perf_dictionary[policy]['acc'].append(list(perfs['accuracy'])[-1])
        perf_dictionary[policy]['energy'].append(list(perfs['tot_energy'])[-1])
        perf_dictionary[policy]['time'].append(list(perfs['tot_time'])[-1])
        perf_dictionary[policy]['acc_list'].append(accs_list)
    print()
    os.makedirs('plots/energy_budget', exist_ok=True)

    index_best_random = perf_dictionary['random']['acc'].index(max(perf_dictionary['random']['acc']))
    index_worst_random = perf_dictionary['random']['acc'].index(min(perf_dictionary['random']['acc']))
    index_best_enea = perf_dictionary['energy_aware']['acc'].index(max(perf_dictionary['energy_aware']['acc']))
    index_worst_enea = perf_dictionary['energy_aware']['acc'].index(min(perf_dictionary['energy_aware']['acc']))

    random_best = perf_dictionary['random']['acc_list'][index_best_random]
    random_worst = perf_dictionary['random']['acc_list'][index_worst_random]
    enea_best = perf_dictionary['energy_aware']['acc_list'][index_best_enea]
    enea_worst = perf_dictionary['energy_aware']['acc_list'][index_worst_enea]

    fig, ax = plt.subplots()
    ax.plot(random_best, c='#66c2a5', marker='s', label="Random best")
    ax.plot(random_worst, c='#fc8d62', marker='o', label="Random worst")
    ax.plot(enea_best, c='#8da0cb', marker='^', label="EneA-FL best")
    ax.plot(enea_worst, c='#e78ac3', marker='v', label="EneA-FL worst")
    # inset axes....
    axins = ax.inset_axes([0.6, 0.5, 0.35, 0.3])
    axins.plot(random_best, c='#66c2a5', marker='s', label="Random best")
    axins.plot(random_worst, c='#fc8d62', marker='o', label="Random worst")
    axins.plot(enea_best, c='#8da0cb', marker='^', label="EneA-FL best")
    axins.plot(enea_worst, c='#e78ac3', marker='v', label="EneA-FL worst")
    # subregion of the original image
    x1, x2, y1, y2 = 1.5, 8.5, 0.875, 0.975
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.xlabel("Round", size=14)
    plt.ylabel("Accuracy", size=14)
    # plt.ylim(5, 35)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', prop={'size': 14})
    plt.title(r'Enea-FL vs Random given energy budget of 1 MJ', size=16)
    plt.savefig('plots/energy_budget/random_vs_ours.pdf')
    plt.show()

    plt.boxplot([perf_dictionary['random']['acc'],
                 perf_dictionary['energy_aware']['acc']],
                positions=[1, 2],
                labels=['Random', 'EneA-FL'])
    # plt.xlabel("Round", size=14)
    plt.ylabel("Accuracy", size=14)
    # plt.ylim(0.92, 1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(r'Enea-FL vs Random given energy budget of 1 MJ', size=16)
    plt.savefig('plots/energy_budget/random_vs_ours_box.pdf')
    plt.show()


def plot_time_budget():
    root_dir = os.path.join(BASE_DIR, 'time_budget_all_acc_target_10_runs_each', 'metrics')
    sub_dirs = [x[0] for x in os.walk(root_dir)][1:]
    sim_ids = [dire.split('/')[-1] for dire in sub_dirs]
    print(sim_ids)
    policies = ['random', 'energy_aware']
    perf_dictionary = {pol: {'acc': [],
                             'energy': [],
                             'time': [],
                             'acc_list': []} for pol in policies}
    for sim_id in sim_ids:
        args_file = os.path.join(BASE_DIR, 'time_budget_all_acc_target_10_runs_each', 'metrics', sim_id, 'args.txt')
        with open(args_file) as f:
            sim_arguments = f.read()
            sim_arguments = json.loads(sim_arguments)
        policy = sim_arguments['policy']
        perf_file = os.path.join(BASE_DIR, 'time_budget_all_acc_target_10_runs_each',
                                 'metrics', sim_id, 'final_metrics.csv')
        print(sim_id)
        perfs = pd.read_csv(perf_file)
        accs_list = list(perfs['accuracy'])
        if accs_list[-2] < 0.9:
            continue
        perf_dictionary[policy]['acc'].append(list(perfs['accuracy'])[-1])
        perf_dictionary[policy]['energy'].append(list(perfs['tot_energy'])[-1])
        perf_dictionary[policy]['time'].append(list(perfs['tot_time'])[-1])
        perf_dictionary[policy]['acc_list'].append(accs_list)
    print()
    os.makedirs('plots/time_budget', exist_ok=True)

    index_best_random = perf_dictionary['random']['acc'].index(max(perf_dictionary['random']['acc']))
    index_worst_random = perf_dictionary['random']['acc'].index(min(perf_dictionary['random']['acc']))
    index_best_enea = perf_dictionary['energy_aware']['acc'].index(max(perf_dictionary['energy_aware']['acc']))
    index_worst_enea = perf_dictionary['energy_aware']['acc'].index(min(perf_dictionary['energy_aware']['acc']))

    random_best = perf_dictionary['random']['acc_list'][index_best_random]
    random_worst = perf_dictionary['random']['acc_list'][index_worst_random]
    enea_best = perf_dictionary['energy_aware']['acc_list'][index_best_enea]
    enea_worst = perf_dictionary['energy_aware']['acc_list'][index_worst_enea]

    fig, ax = plt.subplots()
    ax.plot(random_best, c='#66c2a5', marker='s', label="Random best")
    ax.plot(random_worst, c='#fc8d62', marker='o', label="Random worst")
    ax.plot(enea_best, c='#8da0cb', marker='^', label="EneA-FL best")
    ax.plot(enea_worst, c='#e78ac3', marker='v', label="EneA-FL worst")
    # inset axes....
    axins = ax.inset_axes([0.6, 0.5, 0.35, 0.3])
    axins.plot(random_best, c='#66c2a5', marker='s', label="Random best")
    axins.plot(random_worst, c='#fc8d62', marker='o', label="Random worst")
    axins.plot(enea_best, c='#8da0cb', marker='^', label="EneA-FL best")
    axins.plot(enea_worst, c='#e78ac3', marker='v', label="EneA-FL worst")
    # subregion of the original image
    x1, x2, y1, y2 = 1.5, 8.5, 0.875, 0.975
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels([])
    axins.set_yticklabels([])

    ax.indicate_inset_zoom(axins, edgecolor="black")
    plt.xlabel("Round", size=14)
    plt.ylabel("Accuracy", size=14)
    # plt.ylim(5, 35)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='lower right', prop={'size': 14})
    plt.title(r'Enea-FL vs Random given time budget of 8 hours', size=16)
    plt.savefig('plots/time_budget/random_vs_ours.pdf')
    plt.show()

    plt.boxplot([perf_dictionary['random']['acc'],
                 perf_dictionary['energy_aware']['acc']],
                positions=[1, 2],
                labels=['Random', 'EneA-FL'])
    # plt.xlabel("Round", size=14)
    plt.ylabel("Accuracy", size=14)
    # plt.ylim(0.92, 1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(r'Enea-FL vs Random given time budget of 8 hours', size=16)
    plt.savefig('plots/time_budget/random_vs_ours_box.pdf')
    plt.show()


def get_perf_files(directory):
    subdirs = [x[0] for x in os.walk(directory)]
    subdirs.remove(directory)
    files_dict = {}
    for subdir in subdirs:
        files_dict[subdir] = os.path.join(directory, subdir, 'final_metrics.csv')
    return files_dict


if __name__ == '__main__':
    for experiment in EXPERIMENTS:
        plot_experiment(experiment)
