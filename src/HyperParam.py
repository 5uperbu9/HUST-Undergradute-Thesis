def hyperparam(data):
    if data == 1:
        path = 'dermatology'
        batch_size = 6
        cdd_size = 60
        img_size = None
        channel = None
        std = [0.9, 5]
    elif data == 2:
        path = 'penguins'
        batch_size = 6
        cdd_size = 30
        img_size = None
        channel = None
        std = [0.9, 5]
    elif data == 3:
        path = 'tomato_leaf/'
        batch_size = 64
        cdd_size = 512
        img_size = 64
        channel = 3
        std = [0.8, 20]
    else:
        path = 'MNIST/'
        batch_size = 16
        cdd_size = 256
        img_size = 28
        channel = 1
        std = [0.9, 15]

    print(path)

    return path, batch_size, cdd_size, img_size, channel, std
