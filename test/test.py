import numpy as np
import sys
import os

sys.path.append('../')
from network import Network
from function import act_dict, init_dict, loss_dict
from optimizer import GradientDescent
import data_handler


def accuracy(net, x_set, y_set):

    len_set = len(x_set)
    corr_count = 0

    for i, entry in enumerate(x_set):

        res = net.forward(np.asarray(x_set[i]).flatten())
        thresh_res = 1 if res.item() >= 0.5 else 0

        if y_set[i].item() == thresh_res:
            corr_count += 1

    return corr_count, len_set


def forward_test():
    in_vec = np.array([3, 3])
    exp_res = np.array([12, 12])

    net = Network(np.array([2, 2, 2]), None, act_dict["identity"],
                  act_dict["identity"], loss_dict["squared"])

    out = net.forward(in_vec)

    return np.equal(out, exp_res).all()


def backward_test():
    in_vec = np.array([3, 3])
    exp_res = np.array([6, 6])

    net = Network(np.array([2, 2, 2]), None, act_dict["identity"],
                  act_dict["identity"], loss_dict["squared"])

    net.forward(in_vec)
    net.backward(exp_res)

    bool_res = True

    for layer in net.layers:
        bool_res = bool_res and np.equal(layer.grad, np.array([36, 36])).all()

    return bool_res


def simple_learning_test_regression():
    in_vec = np.array([3, 3])
    exp_res = np.array([6, 6])

    net = Network(np.array([2, 2, 2]), init_dict["std"], act_dict["sigm"],
                  act_dict["identity"], loss_dict["squared"])

    gd = GradientDescent(net, 0.1, 1, 50)
    gd.optimize(np.asmatrix(in_vec), np.asmatrix(exp_res))

    return net.forward(in_vec)


def simple_learning_test_classification():  # Func: (A or B) xor (C or D)
    train_x = np.matrix('0 0 0 0; 0 1 0 1; 0 0 0 1; 0 1 0 0; 1 0 0 0; 1 0 1 0; 1 1 1 1')
    train_y = np.matrix('0; 0; 1; 1; 1; 0; 0')

    net = Network(np.array([4, 4, 1]),
                  init_dict["norm"],
                  act_dict["tanh"],
                  act_dict["sigm"],
                  loss_dict["nll"], [0.5, 0.5])

    gd = GradientDescent(net, 0.1, 1, 500)
    gd.optimize(train_x, train_y)

    return accuracy(net, train_x, train_y)


def simple_and_learning_test_classification():  # Func: A and D
    train_x = np.matrix('0 0; 0 1; 1 0; 1 1')
    train_y = np.matrix('0; 0; 0; 1')

    net = Network(np.array([2, 1]),
                  init_dict["norm"],
                  act_dict["tanh"],
                  act_dict["sigm"],
                  loss_dict["nll"], [0, 0.5])

    gd = GradientDescent(net, 1, 4, 100)
    gd.optimize(train_x, train_y)

    return accuracy(net, train_x, train_y)


print("Forward test: ", forward_test())
print("Backward test: ", backward_test())
print("Simple regression test: ", simple_learning_test_regression())
print("Simple AND classification test: ", simple_and_learning_test_classification())
print("Simple classification test: ", simple_learning_test_classification())


path_train = os.path.join('..', 'datasets', 'monks-1.train')
path_test = os.path.join('..', 'datasets', 'monks-1.test')

train_x, train_y = data_handler.read_monk(path_train)
test_x, test_y = data_handler.read_monk(path_test)

net = Network(np.array([6, 6, 1]),
              init_dict["norm"],
              act_dict["tanh"],
              act_dict["sigm"],
              loss_dict["nll"], [0, 0.5])

gd = GradientDescent(net, 0.1, train_x.shape[0], 1000)

gd.optimize(train_x, train_y)
print(accuracy(net, train_x, train_y))
print(accuracy(net, test_x, test_y))
