

def tot_samples_dataset(dataset='femnist'):
    if dataset == 'femnist':
        return 1000
    elif dataset == 'sent140':
        return 500
    else:
        raise ValueError('Dataset "{}" is not available'.format(dataset))
