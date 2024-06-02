import random
from Initial_Seeding import initial_seeding
from Data_Process import read_dataset
from BatchAL import batch_al

if __name__ == '__main__':
    data = int(input())
    #  1: dermatology
    #  2: penguins
    #  3: tomato leaf disease
    #  4：MNIST
    
    root = './dataset/'
    if data == 1:
        path = 'dermatology'
        batch_size = 6
        cdd_size = 60
        img_size = None
        channel = None
        std = [0.95, 10]
    elif data == 2:
        path = 'penguins'
        batch_size = 6
        cdd_size = 60
        img_size = None
        channel = None
        std = [0.95, 5]
    elif data == 3:
        path = 'tomato_leaf/'
        batch_size = 64
        cdd_size = 256
        img_size = 64
        channel = 3
        std = [0.8, 20]
    else:
        path = 'MNIST/'
        batch_size = 16
        cdd_size = 256
        img_size = 28
        channel = 1
        ratio = 0.8
        std = [0.9, 15]

    X_train, y_train, X_val, y_val, k = read_dataset(data, root + path, img_size, channel)
    train_size = X_train.shape[0]

    first_batch = first_batch_choose(dsize, batch_size, samples)
    # first_batch = random.sample(range(train_size), batch_size)

    data_size = batch_al(X_train, y_train, X_val, y_val, k, batch_size, cdd_size, first_batch, img_size, channel, std)
    print('data_size = ', data_size)
