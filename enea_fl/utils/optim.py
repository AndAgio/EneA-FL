def get_lr(round_index, original_lr=0.1, decay=0.1):
    if round_index <= 30:
        return original_lr
    elif 30 < round_index <= 60:
        return original_lr*decay
    if round_index > 60:
        return original_lr*decay*decay
