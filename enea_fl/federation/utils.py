"""Writes the given metrics in a csv."""

import numpy as np
import os
import pandas as pd
import sys

models_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(models_dir)

COLUMN_NAMES = ['client_id', 'round_number', 'num_samples', 'set']


def _print_metrics(
        round_number,
        client_ids,
        metrics,
        num_samples,
        partition,
        metrics_dir,
        metrics_name):
    """Prints or appends the given metrics in a csv.

    The resulting dataframe is of the form:
        client_id, round_number, hierarchy, num_samples, metric1, metric2
        twebbstack, 0, , 18, 0.5, 0.89

    Args:
        round_number: Number of the round the metrics correspond to. If
            0, then the file in path is overwritten. If not 0, we append to
            that file.
        client_ids: Ids of the clients. Not all ids must be in the following
            dicts.
        metrics: Dict keyed by client id. Each element is a dict of metrics
            for that client in the specified round. The dicts for all clients
            are expected to have the same set of keys.
        num_samples: Dict keyed by client id. Each element is the number of test
            samples for the client.
        partition: String. Value of the 'set' column.
        metrics_dir: String. Directory for the metrics file. May not exist.
        metrics_name: String. Filename for the metrics file. May not exist.
    """
    os.makedirs(metrics_dir, exist_ok=True)
    path = os.path.join(metrics_dir, '{}.csv'.format(metrics_name))

    columns = COLUMN_NAMES + get_metrics_names(metrics)
    client_data = pd.DataFrame(columns=columns)
    for i, c_id in enumerate(client_ids):
        current_client = {
            'client_id': c_id,
            'round_number': round_number,
            'num_samples': num_samples.get(c_id, np.nan),
            'set': partition,
        }

        current_metrics = metrics.get(c_id, {})
        for metric, metric_value in current_metrics.items():
            current_client[metric] = metric_value
        client_data.loc[len(client_data)] = current_client

    mode = 'w' if round_number == 0 else 'a'
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


def get_stat_writer_function(ids, num_samples, metrics_dir, metrics_name):
    def writer_fn(num_round, metrics, partition):
        _print_metrics(
            num_round, ids, metrics, num_samples, partition, metrics_dir,
            '{}_{}'.format(metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(metrics_dir, metrics_name):
    def writer_fn(num_round, ids, metrics, num_samples):
        _print_metrics(
            num_round, ids, metrics, num_samples, 'train', metrics_dir,
            '{}_{}'.format(metrics_name, 'sys'))

    return writer_fn


def print_metrics(metrics, weights, prefix=''):
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
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))


def print_stats(num_round, server, writer, use_val_set):
    train_stat_metrics = server.test_model(set_to_use='train')
    print_metrics(train_stat_metrics, server.get_clients_info(server.get_all_workers())[1], prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(set_to_use=eval_set)
    print_metrics(test_stat_metrics, server.get_clients_info(server.get_all_workers())[1], prefix='{}_'.format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)
