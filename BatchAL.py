from Classifiers import SoftmaxRegression, CNN
from Uncertainty_Calc import Entropy
from Coreset import CoresetGreedy


def batch_al(X_train, y_train, X_val, y_val, k, batch_size, cdd_size, labelled, img_size, channel, std):
    # info_calc = Entropy(train_size=X_train.shape[0], cdd_size=batch_size)
    info_calc = Entropy(train_size=X_train.shape[0], cdd_size=cdd_size)
    coreset = CoresetGreedy(batch_size=batch_size, cdd_size=cdd_size)

    if not img_size:
        classifier = SoftmaxRegression(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, k=k)
    else:
        classifier = CNN(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, k=k,
                         batch_size=batch_size, img_size=img_size, channel=channel)
    probs, acc = classifier.fit(labelled)

    cnt = 0
    print(cnt, '\t', acc)
    accuracy = [acc]

    while True:
        cnt += 1

        # new_batch = info_calc.cdd_set_choose(labelled, probs)
        cdd_set = info_calc.cdd_set_sampling(labelled, probs)
        new_batch = coreset.batch_choose(labelled, cdd_set, probs)

        labelled += new_batch
        probs, acc = classifier.fit(labelled)

        accuracy.append(acc)
        print(cnt, '\t', acc)

        if len(labelled) != len(set(labelled)):
            print('error')

        if acc >= std[0] and cnt >= std[1]:
            break

    return len(labelled), accuracy
