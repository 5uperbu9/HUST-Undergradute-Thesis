from UncertaintyCalc import UncertaintyCalc
from CoresetGreedy import CoresetGreedy
from Classifiers import SoftmaxRegression, CNN


def batch_al(X_train, y_train, X_val, y_val, n, k, labelled,
             batch_size, cdd_size, img_size, channel, budget):

    # uncertainty = Uncertainty_Calc(train_size=train_size, cdd_size=batch_size)
    uncertainty = UncertaintyCalc(n=n, cdd_size=cdd_size)
    coreset = CoresetGreedy(batch_size=batch_size, cdd_size=cdd_size)

    if not img_size:
        classifier = SoftmaxRegression(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, k=k)
    else:
        classifier = CNN(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, k=k,
                         batch_size=batch_size, img_size=img_size, channel=channel)
    probs, acc = classifier.fit(labelled)

    b = 0
    accuracy = [acc]

    while True:

        # new_batch = uncertainty.cdd_set_sampling(labelled, probs)
        cdd_set = uncertainty.cdd_set_sampling(labelled, probs)
        new_batch = coreset.batch_sampling(labelled, cdd_set, probs)
        labelled += new_batch

        probs, acc = classifier.fit(labelled)
        accuracy.append(acc)

        b += 1
        if b >= budget:
            break

    return accuracy
