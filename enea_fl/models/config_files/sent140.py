class Config(object):
    embs_file = 'enea_fl/models/embs.json'
    num_channels = 100
    kernel_size = [3,4,5]
    output_size = 4
    max_epochs = 15
    lr = 0.3
    batch_size = 64
    max_sen_len = 30
    dropout_keep = 0.8
