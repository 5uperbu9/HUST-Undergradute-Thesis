import numpy as np
from scipy.stats import entropy


class Entropy:

    def __init__(self, train_size, cdd_size):
        self.train_size = train_size
        self.cdd_size = cdd_size

    def cdd_set_sampling(self, labelled, probs_, u=0):

        unlabelled = list(set(range(self.train_size)) - set(labelled))
        unlabelled_length = len(unlabelled)
        if unlabelled_length <= self.cdd_size:
            return unlabelled
        probs = probs_[unlabelled].copy()
        if u == 0:
            max_prob = np.max(probs, axis=1)
            max_idx = np.argmax(probs, axis=1)
            for i in range(unlabelled_length):
                probs[i][max_idx[i]] = 0
            second_max_prob = np.max(probs, axis=1)

            uncertainty = 1 - max_prob + second_max_prob
            del max_prob, max_idx, second_max_prob
        else:
            uncertainty = entropy(probs.T)

        sorted_uncertainty = sorted(enumerate(uncertainty), key=lambda x: x[1], reverse=True)
        cdd_set = [unlabelled[e[0]] for i, e in enumerate(sorted_uncertainty) if i < self.cdd_size]

        del uncertainty, sorted_uncertainty
        return cdd_set
