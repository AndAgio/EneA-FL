"""Writes the given metrics in a csv."""

import numpy as np
import os
import pandas as pd
import sys

models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(models_dir)

COLUMN_NAMES = ['client_id', 'round_number', 'set']


def write_metrics_to_csv(
        num_round,
        ids,
        metrics,
        partition,
        metrics_dir,
        sim_id,
        metrics_name):
    """Prints or appends the given metrics in a csv.

    The resulting dataframe is of the form:
        client_id, round_number, hierarchy, num_samples, metric1, metric2
        twebbstack, 0, , 18, 0.5, 0.89

    Args:
        num_round: Number of the round the metrics correspond to. If
            0, then the file in path is overwritten. If not 0, we append to
            that file.
        ids: Ids of the clients. Not all ids must be in the following
            dicts.
        metrics: Dict keyed by client id. Each element is a dict of metrics
            for that client in the specified round. The dicts for all clients
            are expected to have the same set of keys.
        partition: String. Value of the 'set' column.
        metrics_dir: String. Directory for the metrics file. May not exist.
        metrics_name: String. Filename for the metrics file. May not exist.
    """
    metrics_dir = os.path.join(metrics_dir, sim_id)
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, '{}.csv'.format(metrics_name))

    columns = COLUMN_NAMES + get_metrics_names(metrics)
    client_data = pd.DataFrame(columns=columns)
    for i, c_id in enumerate(ids):
        current_client = {
            'client_id': c_id,
            'round_number': num_round,
            'set': partition,
        }

        current_metrics = metrics.get(c_id, {})
        for metric, metric_value in current_metrics.items():
            current_client[metric] = metric_value
        client_data.loc[len(client_data)] = current_client

    mode = 'w' if num_round == 0 else 'a'
    print_dataframe(client_data, path, mode)


def print_dataframe(df, path, mode='w'):
    """Writes the given dataframe in path as a csv"""
    header = mode == 'w'
    df.to_csv(path, mode=mode, header=header, index=False)


def get_metrics_names(metrics):
    """Gets the names of the metrics.

    Args:
        metrics: Dict keyed by client id. Each element is a dict of metrics
            for that client in the specified round. The dicts for all clients
            are expected to have the same set of keys."""
    if len(metrics) == 0:
        return []
    metrics_dict = next(iter(metrics.values()))
    return list(metrics_dict.keys())


def print_workers_metrics(logger, metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = get_metrics_names(metrics)
    to_ret = None
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        logger.print_it('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))


def print_server_metrics(logger, metrics):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    message = 'server:'
    metric_names = get_metrics_names(metrics)
    for metric in metric_names:
        message += ' {} = {}'.format(metric, metrics['server'][metric])
    logger.print_it(message)


def store_results_to_csv(round_ind, metrics, energy, time_taken, metrics_dir, sim_id):
    metrics_dir = os.path.join(metrics_dir, sim_id)
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, 'final_metrics.csv'.format(sim_id))
    if os.path.exists(path):
        data = pd.read_csv(path, index_col=0)
    else:
        columns = ['round'] + [key for key, _ in metrics.items()] + ['round_energy', 'tot_energy',
                                                                     'round_time', 'tot_time']
        data = pd.DataFrame(columns=columns)
    new_row = {'round': round_ind}
    for key, value in metrics.items():
        new_row[key] = value
    new_row['round_energy'] = [energy]
    new_row['round_time'] = [time_taken]
    try:
        new_row['tot_energy'] = [data['tot_energy'].iat[-1] + energy]
        new_row['tot_time'] = [data['tot_time'].iat[-1] + time_taken]
    except IndexError:
        new_row['tot_energy'] = [energy]
        new_row['tot_time'] = [time_taken]
    data = pd.concat([data, pd.DataFrame(new_row)], ignore_index=True)
    data.to_csv(path)

