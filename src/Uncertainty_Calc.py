import numpy as np
from scipy.stats import entropy


class UncertaintyCalc:

    def __init__(self, n, cdd_size):
        self.n = n
        self.cdd_size = cdd_size

    def cdd_set_sampling(self, labelled, probs_, u=0):

        unlabelled = list(set(range(self.n)) - set(labelled))
        U_size = len(unlabelled)
        if U_size <= self.cdd_size:
            return unlabelled

        probs = probs_[unlabelled].copy()
        if u == 0:
            max_prob = np.max(probs, axis=1)
            max_idx = np.argmax(probs, axis=1)
            for i in range(U_size):
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
