import os
import numpy as np

import utils.helpers
from network import Network
from functions.loss_funcs import loss_dict
from functions.act_funcs import act_dict
from functions.init_funcs import init_dict
from functions.metric_funcs import metr_dict
from optimizer import GradientDescent
from utils.data_handler import read_monk, DataHandler
from utils.hype_search import grid_search, stoch_search, eval_model
from utils.plotter import Plotter
from utils.helpers import save_results_to_csv, result_to_str
from utils.debug_tools import check_gradient_combo
from ensemble import Ensemble


def forward_test():
    train_x = np.array([[3, 3]])
    train_y = np.array([[12, 12]])

    net = Network([2, 2, 2], None, act_dict["identity"],
                  act_dict["identity"], loss_dict["squared"])

    out = net.forward(train_x)

    return np.equal(out, train_y).all()


def backward_test():
    train_x = np.array([[3, 3]])
    train_y = np.array([[6, 6]])

    dict_param_net = {
        'conf_layers': [2, 2, 2],
        'init_func': None,
        'act_func': act_dict["identity"],
        'out_func': act_dict["identity"],
        'loss_func': loss_dict["squared"]
    }

    net = Network(**dict_param_net)

    net.forward(train_x)
    net.backward(train_y)

    check_gradient_combo(dict_param_net, train_x, train_y)

    bool_res = True

    for i, layer in enumerate(net.layers):
        bool_res = bool_res and np.equal(layer.grad_w, np.array([[36, 36]])).all()

        if i == 0:
            bool_res = bool_res and np.equal(layer.grad_b, np.array([[12, 12]])).all()
        elif i == 1:
            bool_res = bool_res and np.equal(layer.grad_b, np.array([[6, 6]])).all()

    return bool_res


def simple_learning_test_regression():
    train_x = np.array([[2, 2, 2], [3, 2, 4], [1, 1, 1], [5, 9, 8]])
    train_y = np.array([[4, 4, 4], [6, 4, 8], [2, 2, 2], [10, 18, 16]])
    train_handler = DataHandler(train_x, train_y)

    dict_param_net = {
        'conf_layers': [3, 3],
        'init_func': init_dict["std"],
        'act_func': act_dict["sigm"],
        'out_func': act_dict["identity"],
        'loss_func': loss_dict["squared"],
        'init_scale': 5
    }

    dict_param_sgd = {
        'lr': 0.005,
        'batch_size': 1,
        'reg_val': 0.000,
        'momentum_val': 0.4,
        'nesterov': True,
        'lr_decay_type': "lin",
        'lr_dec_exp_k': 0.005,
        'lr_dec_lin_tau': 80,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.005,
        'lim_epochs': 100,
        'check_gradient': True
    }

    plotter_reg = Plotter(["lr_curve", "lr", "act_val", "grad_norm", "delta_weights"],
                          [loss_dict["squared"]], 2)

    res = eval_model(dict_param_net, dict_param_sgd, train_handler,
                     loss_dict["squared"], n_folds=0, n_runs=30, plotter=plotter_reg)
#    plotter_reg.plot()

    return res['score_tr'], res["epochs"], res["age"]


def simple_and_learning_test_classification():  # Func: A and B
    train_x = np.asarray(np.matrix('0 0; 0 1; 1 0; 1 1'))
    train_y = np.asarray(np.matrix('0; 0; 0; 1'))
    train_handler = DataHandler(train_x, train_y)

    dict_param_net = {
        'conf_layers': [2, 1],
        'init_func': init_dict["std"],
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"],
        'init_scale': 10
    }

    dict_param_sgd = {
        'lr': 0.3,
        'batch_size': 1,
        'momentum_val': 0.6,
        'reg_val': 0.0000,
        'nesterov': True,
        'lr_decay_type': "lin",
        'lr_dec_exp_k': 0.001,
        'lr_dec_lin_tau': 60,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.1,
        'patient': 5,
        'check_gradient': True
    }

    plotter_and = Plotter(["lr_curve", "lr", "act_val", "grad_norm", "delta_weights"],
                          [loss_dict["nll"], metr_dict["miscl. error"]], 2)

    res = eval_model(dict_param_net, dict_param_sgd, train_handler,
                     metr_dict["miscl. error"], n_folds=0, n_runs=10, plotter=plotter_and)

#    plotter_and.plot()

    return res['score_tr'], res["epochs"], res["age"]


def simple_learning_test_classification():  # Func: (A or B) xor (C or D)
    train_x = np.asarray(
        np.matrix('0 0 0 0; 0 0 0 1; 0 0 1 0; 0 0 1 1; 0 1 0 0; 0 1 0 1; 0 1 1 0; ' +
                  '0 1 1 1; 1 0 0 0; 1 0 0 1; 1 0 1 0; 1 0 1 1; 1 1 0 0; 1 1 0 1; ' +
                  '1 1 1 0; 1 1 1 1'))
    train_y = np.asarray(np.matrix('0; 1; 1; 1; 1; 0; 0; ' +
                                   '0; 1; 0; 0; 0; 1; 0; ' +
                                   '0; 0'))
    train_handler = DataHandler(train_x, train_y)

    dict_param_net = {
        'conf_layers': [4, 3, 1],
        'init_func': init_dict["std"],
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"],
        'batch_norm': True,
        'init_scale': 3,
        'batch_momentum': 0.99
    }

    dict_param_sgd = {
        'lr': 0.4,
        'batch_size': 5,
        'reg_val': 0.002,
        'momentum_val': 0.2,
        'nesterov': False,
        'lr_decay_type': "exp",
        'lr_dec_exp_k': 0.07,
        'lr_dec_lin_tau': 20,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.01,
        'patient': 5,
        'lim_epochs': 400,
        'norm_clipping': 8,
        'check_gradient': False
    }

    dict_param_net_grid = {
        'conf_layers': [[4, 2, 1]],
        'init_func': [init_dict["std"]],
        'act_func': [act_dict["tanh"]],
        'out_func': [act_dict["sigm"]],
        'loss_func': [loss_dict["nll"]],
        'batch_norm': [True],
        'init_scale': range(1, 20, 2)
    }

    dict_param_sgd_grid = {
        'lr': np.logspace(-2, 0, 20),
        'batch_size': [-1, 1, 3, 5, 10],
        'reg_val': np.logspace(-4, 0, 20),
        'reg_type': [2],
        'momentum_val': np.logspace(-2, 0, 20),
        'nesterov': [False, True],
        'lim_epochs': [400],
        'lr_decay_type': ["lin", None],
        'lr_dec_lin_tau': np.logspace(0, 3, 20),
        'stop_crit_type': ['delta_w'],
        'epsilon': np.logspace(-2, 0, 20),
        'patient': [5]
    }

    plotter_xor = Plotter(["lr_curve", "lr", "act_val", "grad_norm", "delta_weights"],
                          [loss_dict["nll"], metr_dict["miscl. error"]], 2)

#    results = stoch_search(dict_param_net_grid, dict_param_sgd_grid,
#                          train_handler, metr_dict["miscl. error"], 1000,
#                          n_folds=0, n_runs=20, plotter=plotter_xor)
#    path = os.path.join('.', 'results')
#    save_results_to_csv(path, results)
#    exit()


    #    train_x = (train_x - np.mean(train_x, axis=0))/np.std(train_x, axis=0)

    res = eval_model(dict_param_net, dict_param_sgd, train_handler,
                     metr_dict["miscl. error"], n_folds=0, n_runs=30, plotter=plotter_xor)
    plotter_xor.plot()

    return res['score_tr'], res['score_val'], res["epochs"], res["age"]


def test_monk1_grid():
    path_train = os.path.join('datasets', 'monks-1.train')
    path_test = os.path.join('datasets', 'monks-1.test')

    train_x, train_y = read_monk(path_train)
    test_x, test_y = read_monk(path_test)

    train_handler = DataHandler(train_x, train_y)
    test_handler = DataHandler(test_x, test_y)

    dict_param_net_grid = {
        'conf_layers': [[17, 4, 1]],
        'init_func': [init_dict["norm"]],
        'act_func': [act_dict["tanh"]],
        'out_func': [act_dict["sigm"]],
        'loss_func': [loss_dict["nll"]]
    }

    dict_param_sgd_grid = {
        'lr': [0.1, 0.5],
        'batch_size': [-1],
        'reg_val': [0, 0.001],
        'reg_type': [2],
        'momentum_val': [0, 0.3],
        'nesterov': [False, True],
        'lim_epochs': [400],
        'lr_decay_type': ["lin", None],
        'lr_dec_lin_tau': [100, 200, 400],
        'stop_crit_type': ['delta_w'],
        'epsilon': [0.05, 0.01],
        'patient': [5]
    }

    results = stoch_search(dict_param_net_grid, dict_param_sgd_grid,
                          train_handler, metr_dict["miscl. error"], 1000,
                          n_folds=5, n_runs=1)

    path = os.path.join('.', 'results', 'best_results.csv')
    save_results_to_csv(path, results)

    return results[0]['score_val'], results[0]["epochs"], results[0]["age"]


def test_monk1():
    path_train = os.path.join('datasets', 'monks-1.train')  # 124 patterns
    path_test = os.path.join('datasets', 'monks-1.test')

    train_x, train_y = read_monk(path_train, norm_data=False)
    test_x, test_y = read_monk(path_test, norm_data=False)

    train_handler = DataHandler(train_x, train_y)
    test_handler = DataHandler(test_x, test_y)

    dict_param_net = {
        'conf_layers': [17, 4, 1],
        'init_func': init_dict["std"],
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"],
        'init_scale': 1,
        'batch_norm': True
    }

    dict_param_sgd = {
        'lr': 0.6,
        'batch_size': 10,
        'reg_val': 0.000001,
        'momentum_val': 0.8,
        'nesterov': False,
        'lr_decay_type': "exp",
        'lr_dec_exp_k': 0.00001,
        'lr_dec_lin_tau': 1000,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.01,
        'patient': 5,
        'lim_epochs': 300
    }

    plotter_m1 = Plotter(["lr_curve", "lr", "act_val", "grad_norm", "delta_weights"],
                         [loss_dict["nll"], metr_dict["miscl. error"]], 2)

    res = eval_model(dict_param_net, dict_param_sgd, train_handler,
                     metr_dict["miscl. error"], n_folds=0, n_runs=50,
                     val_handler=test_handler, plotter=plotter_m1)
    plotter_m1.plot()

    return res['score_tr'], res['score_val'], res["epochs"], res["age"]


def test_monk2():
    path_train = os.path.join('datasets', 'monks-2.train')  # 169 patterns
    path_test = os.path.join('datasets', 'monks-2.test')

    train_x, train_y = read_monk(path_train, norm_data=False)
    test_x, test_y = read_monk(path_test, norm_data=False)

    train_handler = DataHandler(train_x, train_y)
    test_handler = DataHandler(test_x, test_y)

    dict_param_net = {
        'conf_layers': [17, 3, 1],
        'init_func': init_dict["std"],
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"],
        'init_scale': 5
    }

    dict_param_sgd = {
        'lr': 0.4,
        'batch_size': 5,
        'reg_val': 0.00000,
        'momentum_val': 0.5,
        'nesterov': False,
        'lr_decay_type': "exp",
        'lr_dec_exp_k': 0.005,
        'lr_dec_lin_tau': 100,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.01,
        'patient': 10,
        'lim_epochs': 300
    }

    plotter_m2 = Plotter(["lr_curve", "lr", "act_val", "grad_norm", "delta_weights"],
                         [loss_dict["nll"], metr_dict["miscl. error"]], 2)

    res = eval_model(dict_param_net, dict_param_sgd, train_handler,
                     metr_dict["miscl. error"], n_folds=5, n_runs=10, plotter=plotter_m2)
    plotter_m2.plot()

    return res['score_tr'], res['score_val'], res["epochs"], res["age"]


#print("Forward test: ", forward_test())
#print("Backward test: ", backward_test())

#reg_res = simple_learning_test_regression()
#print(f"Simple regression test error: {reg_res}")

#clas1_res = simple_and_learning_test_classification()
#print(f"Simple AND classification test error: {clas1_res}")

clas2_res = simple_learning_test_classification()
print(f"Simple classification test error: {clas2_res}")

# Tests on monk1
#monk1_res = test_monk1()
#print(f"Monk 1 score on validation set error: {monk1_res}")

# Tests on monk2
#monk2_res = test_monk2()
#print(f"Monk 2 score on validation set error: {monk2_res}")
