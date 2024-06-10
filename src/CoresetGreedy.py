import numpy as np
from scipy.spatial.distance import jensenshannon


class CoresetGreedy:
    def __init__(self, batch_size, cdd_size):
        self.batch_size = batch_size
        self.cdd_size = cdd_size

    def JSD(self, center_probs, cdd_probs):
        center_size = center_probs.shape[0]
        jsd = []
        if len(center_probs.shape) == 2:
            for i in range(center_size):
                jsd_ = []
                for j in range(self.cdd_size):
                    jsd_.append(jensenshannon(center_probs[i], cdd_probs[j]))
                jsd.append(jsd_)
        else:
            for i in range(self.cdd_size):
                jsd.append(jensenshannon(center_probs, cdd_probs[i]))
        return jsd

    def k_center_greedy(self, cdd_set, labelled_probs, cdd_probs):

        centers = []
        sim = self.JSD(labelled_probs, cdd_probs)

        while len(centers) < self.batch_size:
            min_dist = np.min(np.array(sim), axis=0)  # 最近的中心点的距离
            max_idx = np.argmax(min_dist)  # 最远的点

            centers.append(cdd_set[max_idx])
            sim.append(self.JSD(cdd_probs[max_idx], cdd_probs))
            del min_dist, max_idx

        return centers

    def batch_sampling(self, labelled, cdd_set, probs):
        new_batch = self.k_center_greedy(cdd_set, probs[labelled], probs[cdd_set])
        return new_batch
