from network import Network
from optimizer import GradientDescent
import itertools as it

def grid_search(train_x, train_y, conf_layer_list, init_func_list, act_func_list,
                out_func_list, loss_func_list, bias_list, lr_list, batch_size_list,
                reg_val_list, reg_type_list, epochs_list):

    conf = [0, 0, 0, 0]

    for i, par in parameters:
        for val in par:

            conf_layer = conf_layer_list[conf[i]]





