import random
import csv
import numpy as np
import pandas as pd
from HyperParam import hyperparam
from DataProcess import read_dataset
from InitialSeeding import initial_seeding
from BatchALTest import batch_al_test
from Classifiers import SoftmaxRegression, CNN

root = './dataset/'
test_root = './test_results/'
epochs = 20


def model_test():
    data = int(input())

    path, batch_size, cdd_size, img_size, channel, std = hyperparam(data)
    X_train, y_train, X_val, y_val, n, k = read_dataset(data, root + path, img_size, channel)

    if data > 2:
        classifier = CNN(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, k=k,
                         batch_size=batch_size, img_size=img_size, channel=channel)
    else:
        classifier = SoftmaxRegression(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, k=k)
    probs, acc = classifier.fit(range(n))
    print(acc)


def initial_test():
    for data in range(1, 5):

        path, batch_size, cdd_size, img_size, channel, std = hyperparam(data)
        X_train, y_train, X_val, y_val, n, k = read_dataset(data, root + path, img_size, channel)
        random_seed = np.array(pd.read_csv(test_root + path + '/random_seed.csv', header=None)).tolist()

        with open(test_root + path + '/initial.csv', 'w', newline='') as file:
            writer = csv.writer(file)

            for e in range(epochs):
                initial_seed = random_seed[e][:batch_size]
                acc1 = batch_al_test(X_train, y_train, X_val, y_val, n, k, initial_seed,
                                     batch_size, cdd_size, img_size, channel, std, 0)

                initial_seed = initial_seeding(data, n, batch_size, X_train)
                acc2 = batch_al_test(X_train, y_train, X_val, y_val, n, k, initial_seed,
                                     batch_size, cdd_size, img_size, channel, std, 0)

                writer.writerow([acc1, acc2])


def method_test():
    for data in range(1, 5):

        path, batch_size, cdd_size, img_size, channel, std = hyperparam(data)
        X_train, y_train, X_val, y_val, n, k = read_dataset(data, root + path, img_size, channel)
        random_seed = np.array(pd.read_csv(test_root + path + '/random_seed.csv', header=None)).tolist()

        with open(test_root + path + '/query_times.csv', 'w', newline='') as file1:
            with open(test_root + path + '/accuracy.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer1 = csv.writer(file1)

                for e in range(epochs):
                    # initial_seed = initial_seeding(data, train_size, batch_size, X_train)
                    initial_seed = random_seed[e]
                    query_times = []
                    for t in range(3):
                        query_time, acc = batch_al_test(X_train, y_train, X_val, y_val, n, k, initial_seed,
                                                        batch_size, cdd_size, img_size, channel, std, t)
                        writer.writerow(acc)
                        query_times.append(query_time)
                    writer1.writerow(query_times)


def uncertainty_test():
    for data in range(3, 5):

        path, batch_size, cdd_size, img_size, channel, std = hyperparam(data)
        X_train, y_train, X_val, y_val, n, k = read_dataset(data, root + path, img_size, channel)
        random_seed = np.array(pd.read_csv(test_root + path + '/random_seed.csv', header=None)).tolist()

        with open(test_root + path + '/uncertainty.csv', 'w', newline='') as file:
            writer = csv.writer(file)

            for e in range(epochs):
                # initial_seed = initial_seeding(data, train_size, batch_size, X_train)
                initial_seed = random_seed[e]
                for u in range(2):
                    query_time, acc = batch_al_test(X_train, y_train, X_val, y_val, n, k, initial_seed,
                                                    batch_size, cdd_size, img_size, channel, std, 1, u)
                    writer.writerow(acc)


def size_test():

    path = 'MNIST/'
    X_train, y_train, X_val, y_val, n, k = read_dataset(4, root + path, 28, 1)
    random_seed = np.array(pd.read_csv(test_root + path + '/random_seed.csv', header=None)).tolist()

    # with open(test_root + path + '/cdd_size.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)
    #
    #     for e in range(1):
    #         for t in range(4):
    #             # initial_seed = initial_seeding(4, train_size, batch_size, X_train)
    #             initial_seed = random_seed[e]
    #             query_time, acc = batch_al_test(X_train, y_train, X_val, y_val, n, k, initial_seed,
    #                                             16, 64 * (2 ** t), 28, 1, [0.9, 15], 2)
    #             writer.writerow(acc)

    with open(test_root + path + '/batch_size .csv', 'w', newline='') as file:
        writer = csv.writer(file)

        for e in range(5):
            for t in range(4):
                # initial_seed = initial_seeding(4, train_size, batch_size, X_train)
                initial_seed = random_seed[e+4]
                query_time, acc = batch_al_test(X_train, y_train, X_val, y_val, n, k, initial_seed,
                                                8 * (2 ** t), 256, 28, 1, [0.9, int(32/(2 ** t))], 2)
                writer.writerow(acc)


if __name__ == '__main__':
    # model_test()
    # initial_test()
    # method_test()
    # uncertainty_test()
    size_test()
