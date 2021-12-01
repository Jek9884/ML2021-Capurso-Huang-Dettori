import os
import numpy as np

from network import Network
from functions.loss_funcs import loss_dict
from functions.act_funcs import act_dict
from functions.init_funcs import init_dict
from functions.metric_funcs import accuracy, miscl_error
from optimizer import GradientDescent
from data_handler import read_monk
from hype_search import grid_search
from debug_tools import plot_learning_curve


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

    out = net.forward(in_vec)
    net.backward(exp_res, out)

    bool_res = True

    for i, layer in enumerate(net.layers):
        bool_res = bool_res and np.equal(layer.grad_w, np.array([[36, 36]])).all()

        if i == 0:
            bool_res = bool_res and np.equal(layer.grad_b, np.array([[12, 12]])).all()
        elif i == 1:
            bool_res = bool_res and np.equal(layer.grad_b, np.array([[6, 6]])).all()

    return bool_res


def simple_learning_test_regression():
    in_vec = np.array([3, 3])
    exp_res = np.array([6, 6])

    net = Network(np.array([2, 2, 2]), init_dict["norm"], act_dict["sigm"],
                  act_dict["identity"], loss_dict["squared"])

    gd = GradientDescent(0.1, -1, epochs=20)
    gd.train(net, np.asmatrix(in_vec), np.asmatrix(exp_res))

    return net.forward(in_vec)


def simple_and_learning_test_classification():  # Func: A and B
    train_x = np.matrix('0 0; 0 1; 1 0; 1 1')
    train_y = np.matrix('0; 0; 0; 1')

    net = Network(np.array([2, 1]),
                  init_dict["norm"],
                  act_dict["tanh"],
                  act_dict["sigm"],
                  loss_dict["nll"])

    gd = GradientDescent(1, -1, epochs=100)
    gd.train(net, train_x, train_y)

    net_pred = net.forward(train_x)
    net_pred[net_pred < 0.5] = 0
    net_pred[net_pred >= 0.5] = 1

    return accuracy(train_y, net_pred)


def simple_learning_test_classification():  # Func: (A or B) xor (C or D)
    train_x = np.matrix('0 0 0 0; 0 1 0 1; 0 0 0 1; 0 1 0 0; 1 0 0 0; 1 0 1 0; 1 1 1 1')
    train_y = np.matrix('0; 0; 1; 1; 1; 0; 0')

    net = Network(np.array([4, 4, 1]),
                  init_dict["norm"],
                  act_dict["tanh"],
                  act_dict["sigm"],
                  loss_dict["nll"])

    gd = GradientDescent(0.1, -1, epochs=100)
    gd.train(net, train_x, train_y)

    net_pred = net.forward(train_x)
    net_pred[net_pred < 0.5] = 0
    net_pred[net_pred >= 0.5] = 1

    return accuracy(train_y, net_pred)


def test_monk(path_train, path_test):

    train_x, train_y = read_monk(path_train)
    test_x, test_y = read_monk(path_test)

    dict_param_net = {
        'conf_layers': [[6, 6, 1]],
        'init_func': [init_dict["norm"]],
        'act_func': [act_dict["tanh"]],
        'out_func': [act_dict["sigm"]],
        'loss_func': [loss_dict["nll"]]
    }

    dict_param_sgd = {
        'lr': [0.01, 0.2, 0.5, 0.8],
        'batch_size': [-1],
        'reg_val': [0, 0.001, 0.01, 0.1,],
        'reg_type': [2],
        'momentum_val': [0, 0.01, 0.1],
        'nesterov': [False, True],
        'epochs': [20]
    }

    best_result, best_combo = grid_search(train_x, train_y,
                                          dict_param_net, dict_param_sgd, 10, accuracy)
    print("Best accuracy score (train): ", best_result)
    print(best_combo)

    net = Network(**best_combo[0])
    gd = GradientDescent(**best_combo[1])
    plot_learning_curve(net, gd, train_x, train_y, 100, 1, loss_dict["nll"])

    return best_result


print("Forward test: ", forward_test())
print("Backward test: ", backward_test())
print("Simple regression test: ", simple_learning_test_regression())
print("Simple AND classification test: ", simple_and_learning_test_classification())
print("Simple classification test: ", simple_learning_test_classification())

# Tests on monk1
path_train_monk1 = os.path.join('datasets', 'monks-1.train')
path_test_monk1 = os.path.join('datasets', 'monks-1.test')
print("Monk 1 accuracy on train set:", test_monk(path_train_monk1, path_test_monk1))
