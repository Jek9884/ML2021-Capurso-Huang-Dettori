import os
import numpy as np

from network import Network
from functions.loss_funcs import loss_dict, error_dict
from functions.act_funcs import act_dict
from functions.init_funcs import init_dict
from functions.metric_funcs import metr_dict
from optimizer import GradientDescent
from utils.data_handler import read_monk
from utils.hype_search import grid_search, eval_model
from utils.debug_tools import Plotter

plotter = Plotter(["lr_curve", "lr", "act_val", "grad_norm"],
                  [error_dict["nll"], metr_dict["miscl. error"]], 2)

def forward_test():
    in_vec = np.array([[3, 3]])
    exp_res = np.array([[12, 12]])

    net = Network([2, 2, 2], None, act_dict["identity"],
                  act_dict["identity"], loss_dict["squared"])

    out = net.forward(in_vec)

    return np.equal(out, exp_res).all()


def backward_test():
    in_vec = np.array([[3, 3]])
    exp_res = np.array([[6, 6]])

    net = Network([2, 2, 2], None, act_dict["identity"],
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
    train_x = np.array([[3, 3]])
    train_y = np.array([[6, 6]])

    dict_param_net = {
        'conf_layers': [2, 2, 2],
        'init_func': init_dict["norm"],
        'act_func': act_dict["sigm"],
        'out_func': act_dict["identity"],
        'loss_func': loss_dict["squared"]
    }

    dict_param_sgd = {
        'lr': 0.5,
        'batch_size': -1,
        'momentum_val': 0,
        'nesterov': False,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.01
    }

#    res = eval_model(dict_param_net, dict_param_sgd, train_x, train_y,
#                     error_dict["squared"], plot_bool=True, n_folds=0, n_runs=100)

    net = Network(**dict_param_net)
    gd = GradientDescent(**dict_param_sgd)

    gd.train(net, train_x, train_y)

    return net.forward(train_x), gd.epoch_count


def simple_and_learning_test_classification():  # Func: A and B
    train_x = np.asarray(np.matrix('0 0; 0 1; 1 0; 1 1'))
    train_y = np.asarray(np.matrix('0; 0; 0; 1'))

    dict_param_net = {
        'conf_layers': [2, 1],
        'init_func': init_dict["norm"],
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"]
    }

    dict_param_sgd = {
        'lr': 1,
        'batch_size': 1,
        'momentum_val': 0.8,
        'nesterov': False,
        'lr_decay': True,
        'lr_decay_tau': 50,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.09
    }

#    res = eval_model(dict_param_net, dict_param_sgd, train_x, train_y,
#                     metr_dict["miscl. error"], plot_bool=True, n_folds=0, n_runs=100)

    net = Network(**dict_param_net)
    gd = GradientDescent(**dict_param_sgd)

    gd.train(net, train_x, train_y)

    net_pred = net.forward(train_x)
    net_pred[net_pred < 0.5] = 0
    net_pred[net_pred >= 0.5] = 1

    return metr_dict["accuracy"](train_y, net_pred), gd.epoch_count


def simple_learning_test_classification():  # Func: (A or B) xor (C or D)
    train_x = np.asarray(
        np.matrix('0 0 0 0; 0 1 0 1; 0 0 0 1; 0 1 0 0; 1 0 0 0; 1 0 1 0; 1 1 1 1'))
    train_y = np.asarray(np.matrix('0; 0; 1; 1; 1; 0; 0'))

    dict_param_net = {
        'conf_layers': [4, 2, 1],
        'init_func': init_dict["norm"],
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"]
    }

    dict_param_sgd = {
        'lr': 1,
        'batch_size': 1,
        'reg_val': 0,
        'reg_type': 2,
        'momentum_val': 0.8,
        'nesterov': True,
        'lr_decay': True,
        'lr_decay_tau': 50,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.06,
        'patient': 5
    }

#    res = eval_model(dict_param_net, dict_param_sgd, train_x, train_y,
#                     metr_dict["miscl. error"], plot_bool=True, n_folds=0, n_runs=500)

    net = Network(**dict_param_net)
    gd = GradientDescent(**dict_param_sgd)

    gd.train(net, train_x, train_y)

    net_pred = net.forward(train_x)
    net_pred[net_pred < 0.5] = 0
    net_pred[net_pred >= 0.5] = 1

    return metr_dict["accuracy"](train_y, net_pred), gd.epoch_count


def test_monk1_grid():
    path_train = os.path.join('datasets', 'monks-1.train')
    path_test = os.path.join('datasets', 'monks-1.test')
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
        'lr': [0.1, 0.5],
        'batch_size': [-1, 1, 20],
        'reg_val': [0, 0.5],
        'reg_type': [2],
        'momentum_val': [0, 0.5, 0.99],
        'nesterov': [False, True],
        'lim_epochs': [300],
        'lr_decay': [True, False],
        'lr_decay_tau': [100, 200, 400],
        'stop_crit_type': ['delta_w'],
        'epsilon': [0.001, 0.01],
        'patient': [10]
    }

    metric = error_dict["nll"]
    best_result, best_combo, all_res = grid_search(dict_param_net, dict_param_sgd,
                                                   train_x, train_y, metric,
                                                   n_folds=5, n_runs=1)

    print(f"Best {metric.name} score (train): ", best_result)

    print('init_func: ' + best_combo[0]['init_func'].name)
    print('act_func: ' + best_combo[0]['act_func'].name)
    print('out_func: ' + best_combo[0]['out_func'].name)
    print('loss_func: ' + best_combo[0]['loss_func'].name)
    print(best_combo[1])

    print([res[0] for res in all_res])
    print([res[1] for res in all_res])
    print([res[3] for res in all_res])

    return best_result

def test_monk1():
    path_train = os.path.join('datasets', 'monks-1.train')
    path_test = os.path.join('datasets', 'monks-1.test')
    train_x, train_y = read_monk(path_train, norm_data=False)
    test_x, test_y = read_monk(path_test, norm_data=False)

    dict_param_net = {
        'conf_layers': [6, 6, 1],
        'init_func': init_dict["norm"],
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"]
    }

    dict_param_sgd = {
        'lr': 0.1,
        'batch_size': 20,
        'reg_val': 0.0000001,
        'reg_type': 2,
        'momentum_val': 0.3,
        'nesterov': True,
        'lr_decay': True,
        'lr_decay_tau': 50,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.01,
        'patient': 10
    }

    metric1 = error_dict["nll"]
    metric2 = metr_dict["miscl. error"]

    res = eval_model(dict_param_net, dict_param_sgd, train_x, train_y,
                     metric2, plot_bool=True, n_folds=5, n_runs=10)

    net = Network(**dict_param_net)
    gd = GradientDescent(**dict_param_sgd)

    gd.train(net, train_x, train_y, plotter=plotter)
    plotter.plot()

    net_pred = net.forward(train_x)
    net_pred[net_pred < 0.5] = 0
    net_pred[net_pred >= 0.5] = 1

    return res[1], gd.epoch_count

def test_monk2():
    path_train = os.path.join('datasets', 'monks-2.train')
    path_test = os.path.join('datasets', 'monks-2.test')
    train_x, train_y = read_monk(path_train, norm_data=False)
    test_x, test_y = read_monk(path_test, norm_data=False)

    dict_param_net = {
        'conf_layers': [6, 20, 20, 1],
        'init_func': init_dict["norm"],
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"]
    }

    dict_param_sgd = {
        'lr': 0.5,
        'batch_size': -1,
        'reg_val': 0.0001,
        'reg_type': 2,
        'momentum_val': 0.99,
        'nesterov': False,
        'lr_decay': False,
        'lr_decay_tau': 400,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.01,
        'patient': 10,
        'lim_epochs': 500
    }

    metric1 = error_dict["nll"]
    metric2 = metr_dict["miscl. error"]

    res = eval_model(dict_param_net, dict_param_sgd, train_x, train_y,
                     metric2, plot_bool=True, n_folds=5, n_runs=5)

    net = Network(**dict_param_net)
    gd = GradientDescent(**dict_param_sgd)

    plotter = Plotter(["lr_curve", "lr", "act_val", "grad_norm"],
                      [metric1, metric2], 2)
    gd.train(net, train_x, train_y, plotter=plotter)
    plotter.plot()

    net_pred = net.forward(train_x)
    net_pred[net_pred < 0.5] = 0
    net_pred[net_pred >= 0.5] = 1

    return res[1], gd.epoch_count


print("Forward test: ", forward_test())
print("Backward test: ", backward_test())

reg_res = simple_learning_test_regression()
print(f"Simple regression test: {reg_res[0]}, epochs: {reg_res[1]}")

clas1_res = simple_and_learning_test_classification()
print(f"Simple AND classification test: {clas1_res[0]}, epochs: {clas1_res[1]}")

clas2_res = simple_learning_test_classification()
print(f"Simple classification test: {clas2_res[0]}, epochs: {clas2_res[1]}")

# Tests on monk1
monk1_res = test_monk1()
print(f"Monk 1 score on validation set: {monk1_res}")
exit()

# Tests on monk2
monk2_res = test_monk2()
print(f"Monk 2 score on validation set: {monk2_res}")
