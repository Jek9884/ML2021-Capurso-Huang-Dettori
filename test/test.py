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


# print("Forward test:", forward_test())
# print("Backward test:", backward_test())

net = Network(np.array([6, 6, 6, 1]), init_dict["norm"], act_dict["sigm"], act_dict["identity"], loss_dict["squared"], 0.5)
gd = GradientDescent(net, 0.1, 1, 100)

path = os.path.join('..', 'datasets', 'monks-1.train')
train_x, train_y = data_handler.read_monk(path)
print(train_x)
print(train_y)

gd.optimize(train_x, train_y)
print("\n\n\n")
print(net)
print(net.forward(np.array([1, 1, 1, 1, 1, 1])))  # Exp result = 1
