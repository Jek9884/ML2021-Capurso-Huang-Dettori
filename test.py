import os
import numpy as np

import utils.helpers
from network import Network
from functions.loss_funcs import loss_dict, error_dict
from functions.act_funcs import act_dict
from functions.init_funcs import init_dict
from functions.metric_funcs import metr_dict
from optimizer import GradientDescent
from utils.data_handler import read_monk
from utils.hype_search import grid_search, eval_model
from utils.plotter import Plotter
from utils.helpers import save_results_to_csv, result_to_str
from utils.debug_tools import check_gradient_net

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

    dict_param_net = {
        'conf_layers': [2, 2, 2],
        'init_func': None,
        'act_func': act_dict["identity"],
        'out_func': act_dict["identity"],
        'loss_func': loss_dict["squared"]
    }

    net = Network(**dict_param_net)

#    check_gradient_net(dict_param_net, in_vec, exp_res)
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
    train_x = np.array([[2, 2, 2], [3, 2, 4], [1, 1, 1]])
    train_y = np.array([[4, 4, 4], [6, 4, 8], [2, 2, 2]])

    dict_param_net = {
        'conf_layers': [3, 3],
        'init_func': init_dict["std"],
        'act_func': act_dict["sigm"],
        'out_func': act_dict["identity"],
        'loss_func': loss_dict["squared"],
        'init_scale': 5
    }

    dict_param_sgd = {
        'lr': 0.1,
        'batch_size': -1,
        'momentum_val': 0,
        'nesterov': False,
        'lr_decay_type': None,
        'lr_dec_lin_tau': 20,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.05
    }

    res = eval_model(dict_param_net, dict_param_sgd, train_x, train_y,
                     error_dict["squared"], n_folds=0, n_runs=100)

    return res['score_tr'], res["epochs"]


def simple_and_learning_test_classification():  # Func: A and B
    train_x = np.asarray(np.matrix('0 0; 0 1; 1 0; 1 1'))
    train_y = np.asarray(np.matrix('0; 0; 0; 1'))

    dict_param_net = {
        'conf_layers': [2, 1],
        'init_func': init_dict["std"],
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"]
    }

    dict_param_sgd = {
        'lr': 1,
        'batch_size': 1,
        'momentum_val': 0.9,
        'nesterov': False,
        'lr_decay_type': "lin",
        'lr_dec_exp_k': 0.21,
        'lr_dec_lin_tau': 15,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.02,
        'patient': 5
    }

    res = eval_model(dict_param_net, dict_param_sgd, train_x, train_y,
                     metr_dict["miscl. error"], n_folds=0, n_runs=100)

    return res['score_tr'], res["epochs"]


def simple_learning_test_classification():  # Func: (A or B) xor (C or D)
    train_x = np.asarray(
        np.matrix('0 0 0 0; 0 1 0 1; 0 0 0 1; 0 1 0 0; 1 0 0 0; 1 0 1 0; 1 1 1 1'))
    train_y = np.asarray(np.matrix('0; 0; 1; 1; 1; 0; 0'))

    dict_param_net = {
        'conf_layers': [4, 2, 1],
        'init_func': None,
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"],
        'init_scale': 100
    }

    dict_param_sgd = {
        'lr': 0.99,
        'batch_size': 1,
        'reg_val': 0.005,
        'reg_type': 2,
        'momentum_val': 0.7,
        'nesterov': False,
        'lr_decay_type': "exp",
        'lr_dec_exp_k': 0.001,
        'lr_dec_lin_tau': 500,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.01,
        'patient': 5,
        'lim_epochs': 100
    }

    res = eval_model(dict_param_net, dict_param_sgd, train_x, train_y,
                     metr_dict["miscl. error"], n_folds=2, n_runs=10, plotter=plotter)
    plotter.plot()

    return res['score_tr'], res["epochs"]


def test_monk1_grid():
    path_train = os.path.join('datasets', 'monks-1.train')
    path_test = os.path.join('datasets', 'monks-1.test')
    train_x, train_y = read_monk(path_train)
    test_x, test_y = read_monk(path_test)

    dict_param_net = {
        'conf_layers': [[6, 4, 1]],
        'init_func': [init_dict["norm"]],
        'act_func': [act_dict["tanh"]],
        'out_func': [act_dict["sigm"]],
        'loss_func': [loss_dict["nll"]]
    }

    dict_param_sgd = {
        'lr': [0.1, 0.5, 0.8],
        'batch_size': [-1, 1, 5, 20],
        'reg_val': [0, 0.00001, 0.0001, 0.001],
        'reg_type': [2],
        'momentum_val': [0, 0.3, 0.5],
        'nesterov': [False, True],
        'lim_epochs': [400],
        'lr_decay_type': ["lin", None],
        'lr_dec_lin_tau': [100, 200, 400],
        'stop_crit_type': ['delta_w'],
        'epsilon': [0.05, 0.01, 0.1],
        'patient': [5, 10, 20]
    }

    metric = error_dict["nll"]
    results = grid_search(dict_param_net, dict_param_sgd,
                          train_x, train_y, metric,
                          n_folds=5, n_runs=20)

    path = os.path.join('.', 'results', 'best_results.csv')
    save_results_to_csv(path, results)

    print(result_to_str(results[0], ', '))

    return results[0]['score_val'], res[0]["epochs"]


def test_monk1():
    exit()
    path_train = os.path.join('datasets', 'monks-1.train')
    path_test = os.path.join('datasets', 'monks-1.test')
    train_x, train_y = read_monk(path_train, norm_data=True)
    test_x, test_y = read_monk(path_test, norm_data=False)

    dict_param_net = {
        'conf_layers': [6, 2, 2, 1],
        'init_func': init_dict["std"],
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"],
        'init_scale': 10
    }


    dict_param_sgd = {
        'lr': 0.99,
        'batch_size': 50,
        'reg_val': 0,
        'reg_type': 2,
        'momentum_val': 0.5,
        'nesterov': True,
        'lr_decay_type': None,
        'lr_decay_lin_tau': 400,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.005,
        'patient': 40,
        'lim_epochs': 500
    }

    metric1 = error_dict["nll"]
    metric2 = metr_dict["miscl. error"]

    res = eval_model(dict_param_net, dict_param_sgd, train_x, train_y,
                     metric2, n_folds=5, n_runs=10, plotter=plotter)
    plotter.plot()

    return res['score_tr'], res["epochs"]


def test_monk2():
    path_train = os.path.join('datasets', 'monks-2.train')
    path_test = os.path.join('datasets', 'monks-2.test')
    train_x, train_y = read_monk(path_train, norm_data=False)
    test_x, test_y = read_monk(path_test, norm_data=False)

    dict_param_net = {
        'conf_layers': [6, 4, 1],
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
        'momentum_val': 0.5,
        'nesterov': False,
        'lr_decay_type': None,
        'lr_decay_lin_tau': 400,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.01,
        'patient': 10,
        'lim_epochs': 500
    }

    metric1 = error_dict["nll"]
    metric2 = metr_dict["miscl. error"]

    res = eval_model(dict_param_net, dict_param_sgd, train_x, train_y,
                     metric2, n_folds=5, n_runs=5)

    return res['score_val'], res["epochs"]

print("Forward test: ", forward_test())
print("Backward test: ", backward_test())

reg_res = simple_learning_test_regression()
print(f"Simple regression test error: {reg_res}")

clas1_res = simple_and_learning_test_classification()
print(f"Simple AND classification test error: {clas1_res}")

clas2_res = simple_learning_test_classification()
print(f"Simple classification test error: {clas2_res}")

# Tests on monk1
monk1_res = test_monk1()
print(f"Monk 1 score on validation set error: {monk1_res}")

# Tests on monk2
monk2_res = test_monk2()
print(f"Monk 2 score on validation set error: {monk2_res}")
