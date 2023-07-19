class Config(object):
    embs_file = 'enea_fl/models/embs.json'
    cnn_num_channels = 100
    cnn_kernel_size = [3, 4, 5]
    lstm_layers = 2
    lstm_hidden = 20
    lstm_bidirectional = False
    output_size = 2
    max_sen_len = 30
    dropout = 0.3
