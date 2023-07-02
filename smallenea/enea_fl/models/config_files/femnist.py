class Config(object):
    image_shape = (1, 28, 28)
    conv_channels = [32, 64]  # [20, 50]
    conv_kernels = [5, 5]  # [3, 3]
    conv_strides = [1, 1]
    lin_channels = [4*4*64, 2048]  # [5*5*50, 500]
    n_classes = 62
