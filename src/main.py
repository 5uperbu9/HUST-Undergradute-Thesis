import random
from HyperParam import hyperparam
from DataProcess import read_dataset
from InitialSeeding import initial_seeding
from BatchAL import batch_al

if __name__ == '__main__':
    # 1: dermatology
    # 2: penguins
    # 3: tomato leaf disease
    # 4: MNIST
    data = int(input())
    root = './dataset/'

    path, batch_size, cdd_size, img_size, channel, std = hyperparam(data)
    X_train, y_train, X_val, y_val, n, k = read_dataset(data, root + path, img_size, channel)
    initial_seed = initial_seeding(data, n, batch_size, X_train)
    # initial_seed = random.sample(range(train_size), batch_size)

    accuracy = batch_al(X_train, y_train, X_val, y_val, n, k, initial_seed,
                        batch_size, cdd_size, img_size, channel, std[1])
    print('accuracy = ', accuracy)
