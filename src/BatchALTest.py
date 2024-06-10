import random
from UncertaintyCalc import UncertaintyCalc
from CoresetGreedy import CoresetGreedy
from Classifiers import SoftmaxRegression, CNN


def batch_al_test(X_train, y_train, X_val, y_val, n, k, initial_seed,
                  batch_size, cdd_size, img_size, channel, std, test, u=0):

    labelled = initial_seed.copy()

    if test == 0:
        pass
    elif test == 1:
        info_calc = UncertaintyCalc(n=n, cdd_size=batch_size)
    else:
        info_calc = UncertaintyCalc(n=n, cdd_size=cdd_size)
        coreset = CoresetGreedy(batch_size=batch_size, cdd_size=cdd_size)

    if not img_size:
        classifier = SoftmaxRegression(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, k=k)
    else:
        classifier = CNN(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, k=k,
                         batch_size=batch_size, img_size=img_size, channel=channel)
    probs, acc = classifier.fit(labelled)

    # initial test
    # return acc

    iter_cnt = 0
    query_time = None
    accuracy = [acc]
    print(iter_cnt, '\t', acc)

    while True:

        if test == 0:
            new_batch = random.sample(list(set(range(n)) - set(labelled)), batch_size)
        elif test == 1:
            new_batch = info_calc.cdd_set_sampling(labelled, probs, u=u)
        else:
            cdd_set = info_calc.cdd_set_sampling(labelled, probs, u=u)
            new_batch = coreset.batch_sampling(labelled, cdd_set, probs)
        # print(new_batch)
        del probs

        labelled += new_batch
        probs, acc = classifier.fit(labelled)
        accuracy.append(acc)

        iter_cnt += 1
        print(iter_cnt, '\t', acc)

        if not query_time and acc >= std[0]:
            query_time = len(labelled)
        if query_time and iter_cnt >= std[1]:
            break

    return query_time, accuracy
