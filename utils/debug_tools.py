import numpy as np
from network import Network

# Implementation of gradient checker

"""
    Check the gradient starting from a combo    
    
    Parameters:
        -combo_net_dict: combo to check
        -train_x: train input patterns
        -train_y: train targets
"""


def check_gradient_combo(combo_net_dict, train_x, train_y):
    net = Network(**combo_net_dict)

    # Note: sends entire input training set to the network (batch training)
    for j, _ in enumerate(net.layers):
        check_gradient_layer(net, j, train_x, train_y)


"""
    Check the gradient starting from a network
    
    Parameters:
        -net: network to check
        -train_x: train input patterns
        -train_y: train targets
"""


def check_gradient_net(net, train_x, train_y):
    # Note: sends entire input training set to the network (batch training)
    for j, _ in enumerate(net.layers):
        check_gradient_layer(net, j, train_x, train_y)


"""
    Check the gradient of a specific layer
    
    Parameters:
        -net: network to check
        -layer_idx: idx of the layer to check
        -train_x: train input patterns
        -train_y: train targets
        -eps: scalar used to perturb weights
        -err_tol: max tolerance between analytical and numerical gradient
        -debug_bool: print debug information
"""


def check_gradient_layer(net, layer_idx, train_x, train_y, eps=10 ** -6,
                         err_tol=10 ** -6, debug_bool=False):
    # analytical gradient computation
    net.null_grad()
    net.forward(train_x, training=True)
    net.backward(train_y)
    an_grad_w = net.layers[layer_idx].grad_w
    an_grad_b = net.layers[layer_idx].grad_b
    an_grad_gamma = net.layers[layer_idx].grad_gamma  # none by default
    an_grad_beta = net.layers[layer_idx].grad_beta

    # numerical gradient computation
    backup_w_mat = net.layers[layer_idx].weights  # original weight mat
    backup_b_vec = net.layers[layer_idx].bias  # original bias vec
    backup_gamma_vec = net.layers[layer_idx].batch_gamma
    backup_beta_vec = net.layers[layer_idx].batch_beta

    n_units = backup_w_mat.shape[0]
    n_feat = backup_w_mat.shape[1]

    num_grad_w_mat = []
    num_grad_b_vec = []
    num_grad_gam_vec = []
    num_grad_beta_vec = []

    # perturb wrt all the weights
    for i in range(n_units):

        # weights of a single unit
        unit_vec = []
        for j in range(n_feat):
            num_grad_w = compute_num_grad_mat(net, layer_idx, i, j, train_x,
                                              train_y, backup_w_mat, "weights",
                                              eps)
            unit_vec.append(num_grad_w)

        num_grad_w_mat.append(unit_vec)

        # bias
        num_grad_b = compute_num_grad_vec(net, layer_idx, i, train_x, train_y,
                                          backup_b_vec, "bias", eps)

        num_grad_b_vec.append(num_grad_b)

        # batch normalisation
        if net.batch_norm:
            # gamma
            num_grad_gam = compute_num_grad_vec(net, layer_idx, i, train_x,
                                                train_y, backup_gamma_vec,
                                                "batch_gamma", eps)
            num_grad_gam_vec.append(num_grad_gam)

            # beta
            num_grad_beta = compute_num_grad_vec(net, layer_idx, i, train_x,
                                                 train_y, backup_beta_vec,
                                                 "batch_beta", eps)
            num_grad_beta_vec.append(num_grad_beta)

    # take average gradients as per error definition
    num_patt = len(train_x)
    num_grad_w_mat = np.divide(num_grad_w_mat, num_patt)
    num_grad_b_vec = np.divide(num_grad_b_vec, num_patt)

    grad_w_diff = np.abs(an_grad_w - num_grad_w_mat)
    grad_b_diff = np.abs(an_grad_b - num_grad_b_vec)

    if debug_bool:
        # weights
        print(f"Layer {layer_idx}")
        print("Numerical grad_w: ")
        print(np.array(num_grad_w_mat))
        print("Analytical grad_w: \n", an_grad_w)

        # bias
        print("Numerical grad_b: ")
        print(np.array(num_grad_b_vec))
        print("Analytical grad_b: \n", an_grad_b)

    # check if analytical and numerical gradient are close
    if (grad_w_diff > err_tol).any() or (grad_b_diff > err_tol).any():
        raise RuntimeError(f"Gradient precision is below threshold {err_tol}")

    if net.batch_norm:
        grad_gamma_diff = np.abs(an_grad_gamma - num_grad_gam_vec)
        grad_beta_diff = np.abs(an_grad_beta - num_grad_beta_vec)

        if (grad_gamma_diff > err_tol).any() or (grad_beta_diff > err_tol).any():
            raise RuntimeError(f"Gradient precision is below threshold {err_tol} (batch norm)")


"""
    Compute numerical gradient w.r.t. an element in the matrix
    
    Parameters:
        -net: network used to compute gradient
        -layer_idx: idx of the layer to use to compute the gradient
        -pos_x: row of the matrix to perturb
        -pos_y: column of the matrix to perturb
        -train_x: input patterns
        -train_y: targets
        -backup_mat: copy of the original matrix
        -prop_str: name of the attribute of the net to perturb
        -eps: scalar used to perturb weights
"""


def compute_num_grad_mat(net, layer_idx, pos_x, pos_y, train_x, train_y,
                         backup_mat, prop_str, eps):
    n_units = backup_mat.shape[0]
    n_feat = backup_mat.shape[1]

    new_mat = np.array(backup_mat)
    eye_mat = np.zeros((n_units, n_feat))
    eye_mat[pos_x, pos_y] = 1

    eps_mat = eps * eye_mat
    pert_mat1 = new_mat + eps_mat
    pert_mat2 = new_mat - eps_mat

    setattr(net.layers[layer_idx], prop_str, pert_mat1)
    net.forward(train_x, training=True)
    f1 = net.eval_loss(train_y)

    setattr(net.layers[layer_idx], prop_str, pert_mat2)
    net.forward(train_x, training=True)
    f2 = net.eval_loss(train_y)

    num_grad_mat = (f1 - f2) / (2 * eps)
    setattr(net.layers[layer_idx], prop_str, backup_mat)

    # Singleton array to scalar
    return num_grad_mat.item()


"""
    Compute numerical gradient w.r.t. an element in the vector

    Parameters:
        -net: network used to compute gradient
        -layer_idx: idx of the layer to use to compute the gradient
        -pos: position of the element to perturb
        -train_x: input patterns
        -train_y: targets
        -backup_vec: copy of the original vector
        -prop_str: name of the attribute of the net to perturb
        -eps: scalar used to perturb weights
"""


def compute_num_grad_vec(net, layer_idx, pos, train_x, train_y, backup_vec,
                         prop_str, eps):
    n_units = backup_vec.shape[0]

    new_vec = np.array(backup_vec, copy=True)
    eye_vec = np.zeros((n_units,))
    eye_vec[pos] = 1

    eps_vec = eps * eye_vec
    pert_vec1 = new_vec + eps_vec
    pert_vec2 = new_vec - eps_vec

    setattr(net.layers[layer_idx], prop_str, pert_vec1)
    net.forward(train_x, training=True)
    f1 = net.eval_loss(train_y)

    setattr(net.layers[layer_idx], prop_str, pert_vec2)
    net.forward(train_x, training=True)
    f2 = net.eval_loss(train_y)

    num_grad_vec = (f1 - f2) / (2 * eps)
    setattr(net.layers[layer_idx], prop_str, backup_vec)

    # Singleton array to scalar
    return num_grad_vec.item()
