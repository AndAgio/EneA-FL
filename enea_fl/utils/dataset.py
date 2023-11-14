

def tot_samples_dataset(dataset='femnist', size='small'):
    if dataset in ['femnist', 'mnist']:
        if size == 'small':
            return 438
        elif size == 'medium':
            return 876
        elif size == 'big':
            return 1314
        else:
            raise ValueError('Size "{}" is not available for dataset "{}"!'.format(size, dataset))
    elif dataset == 'nbaiot':
        if size == 'small':
            return 2071
        elif size == 'medium':
            return 4142
        elif size == 'big':
            return 6212
        else:
            raise ValueError('Size "{}" is not available for dataset "{}"!'.format(size, dataset))
    elif dataset == 'sent140':
        if size == 'small':
            return 291
        elif size == 'medium':
            return 582
        elif size == 'big':
            return 872
        else:
            raise ValueError('Size "{}" is not available for dataset "{}"!'.format(size, dataset))
    else:
        raise ValueError('Dataset "{}" is not available'.format(dataset))
