import numpy as np
import sys
import os

sys.path.append('../')
from network import Network
from function import act_dict, init_dict, loss_dict
from optimizer import GradientDescent
import data_handler


def forward_test():
    in_vec = np.array([3, 3])
    exp_res = np.array([12, 12])

    net = Network(np.array([2, 2, 2]), None, act_dict["identity"], act_dict["identity"], loss_dict["squared"])

    out = net.forward(in_vec)

    return np.equal(out, exp_res).all()


def backward_test():
    in_vec = np.array([3, 3])
    exp_res = np.array([6, 6])

    net = Network(np.array([2, 2, 2]), None, act_dict["identity"], act_dict["identity"], loss_dict["squared"])

    net.forward(in_vec)
    net.backward(exp_res)

    bool_res = True

    for layer in net.layers:
        bool_res = bool_res and np.equal(layer.grad, np.array([36, 36])).all()

    return bool_res


def simple_learning_test():
    in_vec = np.array([3, 3])
    exp_res = np.array([6, 6])

    net = Network(np.array([2, 2, 2]), init_dict["std"], act_dict["sigm"],
                  act_dict["identity"], loss_dict["squared"])

    gd = GradientDescent(net, 0.1, 1, 50)
    gd.optimize(np.asmatrix(in_vec), np.asmatrix(exp_res))

    return net.forward(in_vec)


print("Forward test:", forward_test())
print("Backward test:", backward_test())
print(simple_learning_test())

path = os.path.join('..', 'datasets', 'monks-1.train')
train_x, train_y = data_handler.read_monk(path)

net = Network(np.array([6, 2, 1]),
              init_dict["norm"],
              act_dict["sigm"],
              act_dict["sigm"],
              loss_dict["nll"], 0.5)

gd = GradientDescent(net, 0.01, train_x.shape[0], 50)

gd.optimize(train_x, train_y)
print("\n\n\n")
print(net)
print(net.forward(np.asarray(train_x[0]).flatten()))
print(train_y[0])
