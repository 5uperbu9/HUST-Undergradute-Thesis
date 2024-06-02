import random
from Initial_Seeding import initial_seeding
from Data_Process import read_dataset
from BatchAL import batch_al

if __name__ == '__main__':
    # data = int(input())
    #  1: dermatology
    #  2: penguins
    #  3: tomato_leaf leaves

    data = 4

    root = './dataset/'
    if data == 1:
        path = 'dermatology'
        batch_size = 5
        cdd_size = 20
        img_size = None
        channel = None
        ratio = 0.5
        std = [0.95, 10]
    elif data == 2:
        path = 'penguins'
        batch_size = 3
        cdd_size = 20
        img_size = None
        channel = None
        ratio = 0.5
        std = [0.95, 5]
    elif data == 3:
        path = 'tomato_leaf/'
        batch_size = 64
        cdd_size = 128
        img_size = 64
        channel = 3
        ratio = 0.5
        std = [0.85, 20]
    else:
        path = 'MNIST/'
        batch_size = 32
        cdd_size = 64
        img_size = 28
        channel = 1
        ratio = 0.8
        std = [0.9, 20]

    X_train, y_train, X_val, y_val, k = read_dataset(data, root + path, img_size, channel)
    train_size = X_train.shape[0]

    # first_batch = first_batch_choose(dsize, batch_size, samples, img_size)
    first_batch = random.sample(range(train_size), batch_size)

    data_size = batch_al(X_train, y_train, X_val, y_val, k, batch_size, cdd_size, first_batch, img_size, channel, std)
    print('data_size = ', data_size)
