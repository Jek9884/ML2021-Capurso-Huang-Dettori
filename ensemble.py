import numpy as np
from network import Network
from optimizer import GradientDescent

class Ensemble:

    def __init__(self, combo, comp_func='mode', n_models=100, bagging=False):
        self.combo = combo
        self.comp_func = comp_func
        self.net_vec = []
        self.n_models = n_models
        self.bagging = bagging

    def forward(self, data_x):

        scores = None

        for i, net in enumerate(self.net_vec):
            out = np.transpose(net.forward(data_x))
            out = np.array([out])

            if scores is None:
                scores = out
            else:
                scores = np.vstack((scores, out))

        print("Ensemble")

        if self.comp_func == 'mode':

            scores[scores < 0.5] = 0
            scores[scores >= 0.5] = 1

            scores = scores.astype(int)

            ensemble_res = []

            for i in range(scores.shape[1]):
                val, count = np.unique(scores[:, i], axis=0, return_counts=True)
                max_idx = np.argmax(count)
                ensemble_res.append(val[max_idx])

        elif self.comp_func == 'mean':
            ensemble_res = np.average(scores, axis=0)
        else:
            raise ValueError('Invalid comparison function type')

        return np.transpose(ensemble_res)

    def train(self, train_x, train_y):

        for i in range(self.n_models):
            if self.bagging:
                data_x, data_y = gen_subset(train_x, train_y)
            else:
                data_x = train_x
                data_y = train_y

            net = Network(**self.combo['combo_net'])
            opt = GradientDescent(**self.combo['combo_opt'])

            opt.train(net, data_x, data_y)

            self.net_vec.append(net)


def gen_subset(data_x, data_y):
    subset_idx = np.random.randint(0, len(data_x), size=len(data_x))

    return data_x[subset_idx], data_y[subset_idx]