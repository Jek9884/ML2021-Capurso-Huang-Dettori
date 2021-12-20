import numpy as np
from network import Network
from optimizer import GradientDescent


class Ensemble:

    def __init__(self, combo, comp_func='mode', n_models=100):
        self.combo = combo
        self.comp_func = comp_func
        self.net_vec = []
        self.n_models = n_models

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
            net = Network(**self.combo['combo_net'])
            opt = GradientDescent(**self.combo['combo_opt'])

            opt.train(net, train_x, train_y)

            self.net_vec.append(net)

    def eval_res(self):
        pass
