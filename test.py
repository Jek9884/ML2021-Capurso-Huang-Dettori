import os
import numpy as np

from network import Network
from functions.loss_funcs import loss_dict
from functions.act_funcs import act_dict
from functions.init_funcs import init_dict
from functions.metric_funcs import metr_dict
from utils.data_handler import read_monk, DataHandler
from utils.hype_search import grid_search, stoch_search
from utils.evaluation import ComboEvaluator
from utils.plotter import Plotter
from utils.helpers import geomspace_round, save_results_to_csv
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
        'epsilon': 0.01,
        'lim_epochs': 100,
        'check_gradient': True
    }

    plotter_reg = Plotter(["lr_curve", "lr", "act_val", "grad_norm", "delta_weights"],
                          [loss_dict["squared"]], 2)

    evaluator = ComboEvaluator(dict_param_net, dict_param_sgd, train_handler,
                               loss_dict["squared"], n_folds=0, n_runs=30,
                               plotter=plotter_reg)
    res = evaluator.eval()

    plotter_reg.plot()

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

    evaluator = ComboEvaluator(dict_param_net, dict_param_sgd, train_handler,
                               metr_dict["miscl. error"], n_folds=0, n_runs=10,
                               plotter=plotter_and)

    res = evaluator.eval()

    plotter_and.plot()

    return res['score_tr'], res["epochs"], res["age"]


def simple_learning_test_classification():  # Func: (A or B) xor (C or D)

    train_x = np.asarray(
        np.matrix('0 0 0 0; 0 0 0 1; 0 0 1 0; 0 0 1 1; 0 1 0 0; 0 1 0 1; 0 1 1 0; ' +
                  '0 1 1 1; 1 0 0 0; 1 0 0 1; 1 0 1 0; 1 0 1 1; 1 1 0 0; 1 1 0 1; ' +
                  '1 1 1 0; 1 1 1 1'))
    train_y = np.asarray(np.matrix('0; 1; 1; 1; 1; 0; 0; ' +
                                   '0; 1; 0; 0; 0; 1; 0; ' +
                                   '0; 0'))
    train_handler = DataHandler(train_x, train_y, False)

    dict_param_net = {
        'conf_layers': [4, 3, 1],
        'init_func': init_dict["norm"],
        'act_func': act_dict["tanh"],
        'out_func': act_dict["sigm"],
        'loss_func': loss_dict["nll"]
    }

    dict_param_sgd = {
        'lr': 0.6,
        'batch_size': 8,
        'reg_val': 0.0000,
        'momentum_val': 0.1,
        'nesterov': False,
        'lr_decay_type': "lin",
        'lr_dec_exp_k': 0.07,
        'lr_dec_lin_tau': 500,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.009,
        'patient': 30,
        'lim_epochs': 2000
    }

    dict_param_net_grid = {
        'conf_layers': [[4, 3, 1]],
        'init_func': [init_dict["std"]],
        'act_func': [act_dict["tanh"]],
        'out_func': [act_dict["sigm"]],
        'loss_func': [loss_dict["nll"]],
        'init_scale': range(1, 10, 2),
    }

    dict_param_sgd_grid = {
        'lr': geomspace_round(0.01, 1, 10),
        'batch_size': [-1, 4, 8],
        'reg_val': geomspace_round(0.0001, 1, 5),
        'reg_type': [2],
        'momentum_val': geomspace_round(0.01, 1, 5),
        'nesterov': [False, True],
        'lim_epochs': [1000],
        'lr_decay_type': ["lin", "exp", None],
        'lr_dec_lin_tau': geomspace_round(1, 1000, 5),
        'lr_dec_exp_k': geomspace_round(0.001, 1, 5),
        'stop_crit_type': ['delta_w'],
        'epsilon': geomspace_round(0.001, 1, 10),
        'patient': [10]
    }

    plotter_xor = Plotter(["lr_curve", "lr", "act_val", "grad_norm", "delta_weights"],
                          [loss_dict["nll"], metr_dict["miscl. error"]], 2)

#    results = stoch_search(dict_param_net_grid, dict_param_sgd_grid,
#                           train_handler, metr_dict["miscl. error"], 200,
#                           n_folds=0, n_runs=20, plotter=plotter_xor, topk=20)
#    path = os.path.join('.', 'results', 'xor')
#    save_results_to_csv(path, results)
#    exit()

    evaluator = ComboEvaluator(dict_param_net, dict_param_sgd, train_handler,
                               metr_dict["miscl. error"], n_folds=0, n_runs=30,
                               plotter=plotter_xor)
    res = evaluator.eval()

    plotter_xor.plot()

    return res['score_tr'], res['score_val'], res["epochs"], res["age"]

def test_monk1():
    path_train = os.path.join('datasets', 'monks-1.train')  # 124 patterns
    path_test = os.path.join('datasets', 'monks-1.test')

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
        'init_scale': 10
    }

    dict_param_sgd = {
        'lr': 0.3,
        'batch_size': 64,
        'reg_val': 0.0001,
        'momentum_val': 0.6,
        'nesterov': False,
        'lr_decay_type': "lin",
        'lr_dec_exp_k': 0.01,
        'lr_dec_lin_tau': 500,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.005,
        'patient': 10,
        'lim_epochs': 500
    }

    dict_param_net_grid = {
        'conf_layers': [[17, 3, 1]],
        'init_func': [init_dict["std"]],
        'act_func': [act_dict["tanh"]],
        'out_func': [act_dict["sigm"]],
        'loss_func': [loss_dict["nll"]],
        'init_scale': range(1, 20, 2)
    }

    dict_param_sgd_grid = {
        'lr': geomspace_round(0.01, 1, 10),
        'batch_size': [-1, 4, 8, 16, 32],
        'reg_val': geomspace_round(0.0001, 1, 10),
        'reg_type': [2],
        'momentum_val': geomspace_round(0.01, 1, 10),
        'nesterov': [False, True],
        'lim_epochs': [500],
        'lr_decay_type': ["lin", "exp", None],
        'lr_dec_lin_tau': geomspace_round(1, 1000, 10),
        'lr_dec_exp_k': geomspace_round(0.001, 1, 10),
        'stop_crit_type': ['delta_w'],
        'epsilon': geomspace_round(0.001, 1, 20),
        'patient': [5],
        'check_gradient': [False]
    }

    plotter_m1 = Plotter(["lr_curve", "lr", "act_val", "grad_norm", "delta_weights"],
                         [loss_dict["nll"], metr_dict["miscl. error"]], 2)

#    results = stoch_search(dict_param_net_grid, dict_param_sgd_grid,
#                           train_handler, metr_dict["miscl. error"], 50,
#                           n_folds=5, n_runs=20, plotter=plotter_m1, topk=5,
#                           median_stop=None)
#    path = os.path.join('.', 'results', 'monk1')
#    save_results_to_csv(path, results)
#    exit()

    evaluator = ComboEvaluator(dict_param_net, dict_param_sgd, train_handler,
                               metr_dict["miscl. error"], n_folds=5, n_runs=20,
                               val_handler=None, plotter=plotter_m1)
    res = evaluator.eval()

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
        'init_scale': 10
    }

    dict_param_sgd = {
        'lr': 0.6,
        'batch_size': 15,
        'reg_val': 0.00001,
        'momentum_val': 0.7,
        'nesterov': True,
        'lr_decay_type': "lin",
        'lr_dec_exp_k': 0.1,
        'lr_dec_lin_tau': 80,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.02,
        'patient': 5,
        'lim_epochs': 200
    }

    dict_param_net_grid = {
        'conf_layers': [[17, 3, 1]],
        'init_func': [init_dict["std"]],
        'act_func': [act_dict["tanh"]],
        'out_func': [act_dict["sigm"]],
        'loss_func': [loss_dict["nll"]],
        'init_scale': range(1, 10, 4)
    }

    dict_param_sgd_grid = {
        'lr': geomspace_round(0.01, 1, 5),
        'batch_size': [-1, 8, 16, 32, 64],
        'reg_type': [2],
        'reg_val': geomspace_round(0.0001, 1, 3),
        'momentum_val': geomspace_round(0.01, 1, 3),
        'lim_epochs': [500]
    }

    plotter_m2 = Plotter(["lr_curve", "lr", "act_val", "grad_norm", "delta_weights"],
                         [loss_dict["nll"], metr_dict["miscl. error"]], 2)

#    results = grid_search(dict_param_net_grid, dict_param_sgd_grid,
#                          train_handler, metr_dict["miscl. error"],
#                          n_folds=5, n_runs=5, plotter=plotter_m2)
#    path = os.path.join('.', 'results', 'monk2')
#    save_results_to_csv(path, results)
#    exit()

    evaluator = ComboEvaluator(dict_param_net, dict_param_sgd, train_handler,
                               metr_dict["miscl. error"], n_folds=5, n_runs=30,
                               plotter=plotter_m2)
    res = evaluator.eval()

    plotter_m2.plot()

    return res['score_tr'], res['score_val'], res["epochs"], res["age"]


def test_monk3():
    path_train = os.path.join('datasets', 'monks-3.train')  # 122 patterns
    path_test = os.path.join('datasets', 'monks-3.test')

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
        'init_scale': 10
    }

    dict_param_sgd = {
        'lr': 0.1,
        'batch_size': -1,
        'reg_val': 0.00001,
        'momentum_val': 0.8,
        'nesterov': True,
        'lr_decay_type': None,
        'lr_dec_exp_k': 0.001,
        'lr_dec_lin_tau': 80,
        'stop_crit_type': 'delta_w',
        'epsilon': 0.02,
        'patient': 5,
        'lim_epochs': 200
    }

    dict_param_net_grid = {
        'conf_layers': [[17, 4, 1]],
        'init_func': [init_dict["std"]],
        'act_func': [act_dict["tanh"]],
        'out_func': [act_dict["sigm"]],
        'loss_func': [loss_dict["nll"]],
        'batch_norm': [False],
        'init_scale': range(1, 20, 2),
        'batch_momentum': [0.99]
    }

    dict_param_sgd_grid = {
        'lr': geomspace_round(0.01, 1, 10),
        'batch_size': [-1, 4, 8, 16, 32],
        'reg_val': geomspace_round(0.0001, 1, 10),
        'reg_type': [2],
        'momentum_val': geomspace_round(0.01, 1, 10),
        'nesterov': [False, True],
        'lim_epochs': [200],
        'lr_decay_type': ["lin", "exp", None],
        'lr_dec_lin_tau': geomspace_round(1, 1000, 10),
        'lr_dec_exp_k': geomspace_round(0.001, 1, 10),
        'stop_crit_type': ['delta_w'],
        'epsilon': geomspace_round(0.001, 1, 20),
        'patient': [5],
        'norm_clipping': [4],
        'check_gradient': [False]
    }


    plotter_m3 = Plotter(["lr_curve", "lr", "act_val", "grad_norm", "delta_weights"],
                         [loss_dict["nll"], metr_dict["miscl. error"]], 2)

#    results = stoch_search(dict_param_net_grid, dict_param_sgd_grid,
#                          train_handler, metr_dict["miscl. error"], 2000,
#                          n_folds=5, n_runs=20, plotter=plotter_m3)
#    path = os.path.join('.', 'results', 'monk3')
#    save_results_to_csv(path, results)
#    exit()

    evaluator = ComboEvaluator(dict_param_net, dict_param_sgd, train_handler,
                               metr_dict["miscl. error"], n_folds=5, n_runs=20,
                               plotter=plotter_m3)
    res = evaluator.eval()

    plotter_m3.plot()

    return res['score_tr'], res['score_val'], res["epochs"], res["age"]


#print("Forward test: ", forward_test())
#print("Backward test: ", backward_test())

#reg_res = simple_learning_test_regression()
#print(f"Simple regression error: {reg_res}")

#clas1_res = simple_and_learning_test_classification()
#print(f"Simple AND classification error: {clas1_res}")

#clas2_res = simple_learning_test_classification()
#print(f"Simple classification (xor) error: {clas2_res}")

# Tests on monk1
monk1_res = test_monk1()
print(f"Monk 1 error: {monk1_res}")

# Tests on monk2
#monk2_res = test_monk2()
#print(f"Monk 2 error: {monk2_res}")

# Tests on monk3
#monk3_res = test_monk3()
#print(f"Monk 3 error: {monk3_res}")
