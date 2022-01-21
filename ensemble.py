import numpy as np
from network import Network
from optimizer import GradientDescent

"""
    Ensemble
    
    Parameters:
        -combo: tuple of network and optimizer combo
        -agg_func: aggregation function the result of the ensemble
        -n_models: number of models to train
        -bagging: boolean value to apply bagging ensemble method
    
    Attributes:
        -net_vec: vector of n_models networks 
        -opt_vec: vector of n_models optimizers
"""


class Ensemble:

    def __init__(self, combo, agg_func='mode', n_models=100, bagging=False):
        self.combo = combo
        self.agg_func = agg_func
        self.net_vec = []
        self.opt_vec = []
        self.n_models = n_models
        self.bagging = bagging

        for i in range(self.n_models):
            net = Network(**self.combo['combo_net'])
            opt = GradientDescent(**self.combo['combo_opt'])

            self.net_vec.append(net)
            self.opt_vec.append(opt)

    """
        Forward step for the n_models networks. Returns a matrix of outputs of networks
        
        Parameters:
            -data_x: data used to make the forward step
    """

    def forward(self, data_x):

        scores = None

        # for each network stack its predictions inside the score matrix
        for i, net in enumerate(self.net_vec):
            out = np.transpose(net.forward(data_x))
            out = np.array([out])

            if scores is None:
                scores = out
            else:
                scores = np.vstack((scores, out))

        print("Ensemble")

        if self.agg_func == 'mode':

            scores[scores < 0.5] = 0
            scores[scores >= 0.5] = 1

            scores = scores.astype(int)

            ensemble_res = []

            for i in range(scores.shape[1]):
                # count element frequency of the i-column of predictions matrix and keep the most predicted ones
                val, count = np.unique(scores[:, i], axis=0, return_counts=True)
                max_idx = np.argmax(count)
                ensemble_res.append(val[max_idx])

        elif self.agg_func == 'mean':
            ensemble_res = np.average(scores, axis=0)
        else:
            raise ValueError('Invalid comparison function type')

        return np.transpose(ensemble_res)

    """
        Train all the models
        
        Parameters:
            -train_x: input patterns
            -train_y: targets
    """

    def train(self, train_x, train_y):

        for i in range(self.n_models):
            if self.bagging:
                data_x, data_y = gen_subset(train_x, train_y)
            else:
                data_x = train_x
                data_y = train_y

            net = self.net_vec[i]
            opt = self.opt_vec[i]

            opt.train(net, data_x, data_y)


"""
    Generate a subset of a train_x set with replacement
    
    Parameters:
        -data_x: input patterns
        -data_y: targets
"""


def gen_subset(data_x, data_y):
    subset_idx = np.random.randint(0, len(data_x), size=len(data_x))

    return data_x[subset_idx], data_y[subset_idx]
