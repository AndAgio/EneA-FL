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
EXPERIMENTS = ['alpha_beta']


def plot_experiment(experiment):
    if experiment == 'alpha_beta':
        plot_alpha_beta()


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
            all_avg_energies.append(np.mean([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)]]))
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
            energies_avg['color'].append((ene - min_ene)/(max_ene - min_ene))
    energies_std = {'a': [],
                    'b': [],
                    'size': [],
                    'color': []}
    for alpha in alphas:
        for beta in betas:
            energies_std['a'].append(alpha)
            energies_std['b'].append(beta)
            ene = np.mean([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)]])
            tot = ene + np.std([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)]])
            energies_std['size'].append(tot)
            energies_std['color'].append((ene - min_ene)/(max_ene - min_ene))
    print()

    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    colors = [i/500 for i in np.arange(min_ene, max_ene, 100)]
    norm = Normalize(vmin=min_ene.item()/500, vmax=max_ene.item()/500)
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
            plt.text(energies_avg['a'][i]-0.05, energies_avg['b'][i]-2,
                     '{:.1f}'.format(energies_avg['size'][i]/500), fontsize=12)
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
    comparison['color'].append((random_ene_avg - min_ene)/(max_ene - min_ene))

    worst_ene_avg = np.mean([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0., 100)]])
    worst_ene_std = np.std([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0., 100)]])
    comparison['x'].append(0.5)
    comparison['y'].append(0)
    comparison['avg'].append(worst_ene_avg)
    comparison['std'].append(worst_ene_std)
    comparison['color'].append((worst_ene_avg - min_ene)/(max_ene - min_ene))

    best_ene_avg = np.mean([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0.6, 40)]])
    best_ene_std = np.std([item['energy'] / 2000000 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0.6, 40)]])
    comparison['x'].append(1)
    comparison['y'].append(0)
    comparison['avg'].append(best_ene_avg)
    comparison['std'].append(best_ene_std)
    comparison['color'].append((best_ene_avg - min_ene)/(max_ene - min_ene))

    plt.scatter(comparison['x'], comparison['y'],
                s=comparison['avg'],
                c=cmap(comparison['color']),
                alpha=0.5, )
    # plt.scatter(comparison['x'], comparison['y'],
    #             s=comparison['std'],
    #             c=cmap(comparison['color']),
    #             alpha=0.25, )
    for i in range(len(comparison['x'])):
        plt.text(comparison['x'][i]-0.05, comparison['y'][i]-0.02,
                 '{:.1f}'.format(comparison['avg'][i]/500), fontsize=12)
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
                np.mean([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)] if item['rounds'] is not None]))
    all_avg_rounds.append(np.mean([item['rounds'] * 50 for item in perf_dictionary['random'] if item['rounds'] is not None]))
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
            rounds = np.mean([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(alpha, beta)] if item['rounds'] is not None])
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

    random_rounds_avg = np.mean([item['rounds'] * 50 for item in perf_dictionary['random'] if item['rounds'] is not None])
    random_rounds_std = np.std([item['rounds'] * 50 for item in perf_dictionary['random'] if item['rounds'] is not None])
    comparison['x'].append(0)
    comparison['y'].append(0)
    comparison['avg'].append(random_rounds_avg)
    comparison['std'].append(random_rounds_std)
    comparison['color'].append((random_rounds_avg - min_rounds) / (max_rounds - min_rounds))

    worst_rounds_avg = np.mean([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0., 100)] if item['rounds'] is not None])
    worst_rounds_std = np.std([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0., 100)] if item['rounds'] is not None])
    comparison['x'].append(0.5)
    comparison['y'].append(0)
    comparison['avg'].append(worst_rounds_avg)
    comparison['std'].append(worst_rounds_std)
    comparison['color'].append((worst_rounds_avg - min_rounds) / (max_rounds - min_rounds))

    best_rounds_avg = np.mean([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0.6, 40)] if item['rounds'] is not None])
    best_rounds_std = np.std([item['rounds'] * 50 for item in perf_dictionary['a{:.1f}_b{:.1f}'.format(0.6, 40)] if item['rounds'] is not None])
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


def plot_alpha():
    root_dir = os.path.join(BASE_DIR, 'alphas_all_rounds_target_10_runs_each', 'metrics')
    sub_dirs = [x[0] for x in os.walk(root_dir)][1:]
    sim_ids = [dire.split('/')[-1] for dire in sub_dirs]
    print(sim_ids)
    policies = []
    alphas = []
    for sim_id in sim_ids:
        args_file = os.path.join(BASE_DIR, 'alphas_all_rounds_target_10_runs_each', 'metrics', sim_id, 'args.txt')
        with open(args_file) as f:
            sim_arguments = f.read()
            sim_arguments = json.loads(sim_arguments)
            print(sim_arguments)
            policies.append(sim_arguments['policy'])
            alphas.append(sim_arguments['alpha'])
    policies = list(set(policies))
    alphas = sorted(list(set(alphas)))
    print(policies)
    print(alphas)
    perf_dictionary = {alpha: [] for alpha in alphas}
    if 'random' in policies:
        perf_dictionary['random'] = []
    for sim_id in sim_ids:
        args_file = os.path.join(BASE_DIR, 'alphas_all_rounds_target_10_runs_each', 'metrics', sim_id, 'args.txt')
        with open(args_file) as f:
            sim_arguments = f.read()
            sim_arguments = json.loads(sim_arguments)
        policy = sim_arguments['policy']
        alpha = sim_arguments['alpha']
        perf_file = os.path.join(BASE_DIR, 'alphas_all_rounds_target_10_runs_each',
                                 'metrics', sim_id, 'final_metrics.csv')
        print(sim_id)
        perfs = pd.read_csv(perf_file)
        if policy == 'random':
            perf_dictionary['random'].append({'acc': list(perfs['accuracy'])[-1],
                                              'energy': list(perfs['tot_energy'])[-1],
                                              'time': list(perfs['tot_time'])[-1], })
        else:
            perf_dictionary[alpha].append({'acc': list(perfs['accuracy'])[-1],
                                           'energy': list(perfs['tot_energy'])[-1],
                                           'time': list(perfs['tot_time'])[-1], })
    print()
    xs = alphas
    energies_avg = [np.mean([item['energy'] / 1000000 for item in perf_dictionary[alpha]]) for alpha in alphas]
    energies_std = [np.std([item['energy'] / 1000000 for item in perf_dictionary[alpha]]) for alpha in alphas]
    print()
    times_avg = [np.mean([item['time'] for item in perf_dictionary[alpha]]) for alpha in alphas]
    times_std = [np.std([item['time'] for item in perf_dictionary[alpha]]) for alpha in alphas]
    print()

    plt.plot(xs, energies_avg, 'g-', label="Smart selection")
    plt.fill_between(xs,
                     [energies_avg[i] - energies_std[i] for i in range(len(energies_avg))],
                     [energies_avg[i] + energies_std[i] for i in range(len(energies_avg))],
                     alpha=0.5, edgecolor='g', facecolor='g')
    random_energies_avg = [np.mean([item['energy'] / 1000000 for item in perf_dictionary['random']]) for _ in alphas]
    random_energies_std = [np.std([item['energy'] / 1000000 for item in perf_dictionary['random']]) for _ in alphas]
    plt.plot(xs, random_energies_avg, 'b-', label="Random selection")
    plt.fill_between(xs,
                     [random_energies_avg[i] - random_energies_std[i] for i in range(len(random_energies_avg))],
                     [random_energies_avg[i] + random_energies_std[i] for i in range(len(random_energies_avg))],
                     alpha=0.5, edgecolor='b', facecolor='b')
    plt.xlabel("alpha")
    plt.ylabel("Energy [KJ]")
    plt.ylim(0, 6000)
    plt.legend()
    plt.show()

    plt.plot(xs, times_avg, 'g-', label="Smart selection")
    plt.fill_between(xs,
                     [times_avg[i] - times_std[i] for i in range(len(times_avg))],
                     [times_avg[i] + times_std[i] for i in range(len(times_avg))],
                     alpha=0.5, edgecolor='g', facecolor='g')
    random_times_avg = [np.mean([item['time'] for item in perf_dictionary['random']]) for _ in alphas]
    random_times_std = [np.std([item['time'] for item in perf_dictionary['random']]) for _ in alphas]
    plt.plot(xs, random_times_avg, 'b-', label="Random selection")
    plt.fill_between(xs,
                     [random_times_avg[i] - random_times_std[i] for i in range(len(random_times_avg))],
                     [random_times_avg[i] + random_times_std[i] for i in range(len(random_times_avg))],
                     alpha=0.5, edgecolor='b', facecolor='b')
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Time [s]")
    plt.ylim(0, 150000)
    plt.legend()
    plt.show()

    acc_avg = [np.mean([item['acc'] for item in perf_dictionary[alpha] if item['acc'] > 0.96]) for alpha in alphas]
    acc_std = [np.std([item['acc'] for item in perf_dictionary[alpha] if item['acc'] > 0.96]) for alpha in alphas]
    plt.plot(xs, acc_avg, 'g-', label="Smart selection")
    plt.fill_between(xs,
                     [acc_avg[i] - acc_std[i] for i in range(len(acc_avg))],
                     [acc_avg[i] + acc_std[i] for i in range(len(acc_avg))],
                     alpha=0.5, edgecolor='g', facecolor='g')
    random_acc_avg = [np.mean([item['acc'] for item in perf_dictionary['random'] if item['acc'] > 0.96]) for _ in
                      alphas]
    random_acc_std = [np.std([item['acc'] for item in perf_dictionary['random'] if item['acc'] > 0.96]) for _ in alphas]
    plt.plot(xs, random_acc_avg, 'b-', label="Random selection")
    plt.fill_between(xs,
                     [random_acc_avg[i] - random_acc_std[i] for i in range(len(random_acc_avg))],
                     [random_acc_avg[i] + random_acc_std[i] for i in range(len(random_acc_avg))],
                     alpha=0.5, edgecolor='b', facecolor='b')
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Accuracy")
    # plt.ylim(0, 1)
    plt.legend()
    plt.show()

    fig, host = plt.subplots()
    par1 = host.twinx()
    p1, = host.plot(xs, energies_avg, 'r-', label="Energy")
    p2, = par1.plot(xs, times_avg, 'g-', label="Time")
    host.fill_between(xs,
                      [energies_avg[i] - energies_std[i] for i in range(len(energies_avg))],
                      [energies_avg[i] + energies_std[i] for i in range(len(energies_avg))],
                      alpha=0.5, edgecolor='r', facecolor='r')
    host.fill_between(xs,
                      [times_avg[i] - times_std[i] for i in range(len(times_avg))],
                      [times_avg[i] + times_std[i] for i in range(len(times_avg))],
                      alpha=0.5, edgecolor='g', facecolor='g')
    host.set_ylim(0, 2500)
    par1.set_ylim(0, 55000)
    host.set_xlabel("alpha")
    host.set_ylabel("Energy [KJ]")
    par1.set_ylabel("Time [s]")
    lines = [p1, p2]
    host.legend(lines, [l.get_label() for l in lines])
    for ax in [par1]:
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        plt.setp(ax.spines.values(), visible=False)
        ax.spines["right"].set_visible(True)
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par1.spines["right"].set_edgecolor(p2.get_color())
    host.tick_params(axis='y', colors=p1.get_color())
    par1.tick_params(axis='y', colors=p2.get_color())
    fig.show()


def plot_alpha_beta_best():
    alphas = [i for i in np.arange(0, 1.05, 0.2)]
    betas = [1 - i for i in alphas]
    perf_dictionary = {alpha: None for alpha in alphas}
    for alpha in alphas:
        perf_file = os.path.join(BASE_DIR, 'alphas_gpus_only_acc_target_best', 'metrics',
                                 'a_{:.1f}'.format(alpha), 'final_metrics.csv')
        perfs = pd.read_csv(perf_file)
        perf_dictionary[alpha] = {'acc': list(perfs['accuracy']),
                                  'energy': list(perfs['tot_energy']),
                                  'time': list(perfs['tot_time']), }
    perf_file = os.path.join(BASE_DIR, 'alphas_gpus_only_acc_target_best', 'metrics', 'random', 'final_metrics.csv')
    perfs = pd.read_csv(perf_file)
    perf_dictionary['random'] = {'acc': list(perfs['accuracy']),
                                 'energy': list(perfs['tot_energy']),
                                 'time': list(perfs['tot_time']), }
    # Plot
    xs = alphas
    ys = betas
    energies = [perf_dictionary[alpha]['energy'][-1] / 1000000 for alpha in alphas]
    plt.scatter(xs, ys,
                s=energies,
                alpha=0.5, )
    plt.xlabel("alpha", size=16)
    plt.ylabel("beta", size=16)
    plt.title("Energy spent", size=18)
    plt.show()

    plt.plot(xs, energies)
    plt.xlabel("alpha", size=16)
    plt.ylabel("Energy [KJ]", size=16)
    plt.title("Energy spent", size=18)
    plt.show()

    times = [perf_dictionary[alpha]['time'][-1] for alpha in alphas]
    fig, host = plt.subplots()
    par1 = host.twinx()
    p1, = host.plot(xs, energies, 'r-', label="Energy")
    p2, = par1.plot(xs, times, 'g-', label="Time")
    host.set_ylim(0, 400)
    par1.set_ylim(0, 12000)
    host.set_xlabel("alpha")
    host.set_ylabel("Energy [KJ]")
    par1.set_ylabel("Time [s]")
    lines = [p1, p2]
    host.legend(lines, [l.get_label() for l in lines])
    for ax in [par1]:
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        plt.setp(ax.spines.values(), visible=False)
        ax.spines["right"].set_visible(True)
    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())
    par1.spines["right"].set_edgecolor(p2.get_color())
    host.tick_params(axis='y', colors=p1.get_color())
    par1.tick_params(axis='y', colors=p2.get_color())
    fig.show()

    xs = [i for i in range(1, 16)]
    ys = perf_dictionary['random']['acc']
    plt.plot(ys, label='random')
    for alpha in alphas:
        ys = perf_dictionary[alpha]['acc']
        plt.plot(ys, label='a={:.1f}, b={:.1f}'.format(alpha, 1 - alpha))
    plt.legend()
    plt.show()

    ys = [ene / 1000000 for ene in perf_dictionary['random']['energy']]
    plt.plot(ys, label='random')
    for alpha in alphas:
        ys = [ene / 1000000 for ene in perf_dictionary[alpha]['energy']]
        plt.plot(ys, label='a={:.1f}, b={:.1f}'.format(alpha, 1 - alpha))
    plt.legend()
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
