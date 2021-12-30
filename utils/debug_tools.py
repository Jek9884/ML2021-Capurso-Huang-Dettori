import numpy as np

from network import Network

# Implementation of gradient checker

def check_gradient_combo(combo_net_dict, train_x, train_y):

    net = Network(**combo_net_dict)

    # Note: sends entire input training set to the network (batch training)
    for j, _ in enumerate(net.layers):

        check_gradient_layer(net, j, train_x, train_y)


def check_gradient_net(net, train_x, train_y):

    # Note: sends entire input training set to the network (batch training)
    for j, _ in enumerate(net.layers):

        check_gradient_layer(net, j, train_x, train_y)


def check_gradient_layer(net, layer_idx, train_x, train_y, epsilon=10**-6,
                         err_tol=10**-6, debug_bool=False):

    # Analytical gradient computation
    net.null_grad()
    net.forward(train_x, training=True)
    net.backward(train_y)
    an_grad_w = net.layers[layer_idx].grad_w
    an_grad_b = net.layers[layer_idx].grad_b

    # Numerical gradient computation
    backup_w_mat = net.layers[layer_idx].weights  # Original weight mat
    backup_b_vec = net.layers[layer_idx].bias  # Original bias vec

    n_units = backup_w_mat.shape[0]
    n_feat = backup_w_mat.shape[1]

    num_grad_w_mat = []
    num_grad_b_vec = []

    # Perturbe wrt all the weights
    for i in range(n_units):

        # Weights
        unit_vec = []
        for j in range(n_feat):

            new_weights = np.array(backup_w_mat, copy=True)
            eye_mat = np.zeros((n_units, n_feat))
            eye_mat[i, j] = 1

            eps_mat = epsilon*eye_mat
            pert_mat1 = new_weights + eps_mat
            pert_mat2 = new_weights - eps_mat

            net.layers[layer_idx].weights = pert_mat1
            net.forward(train_x, training=True)
            f1 = net.eval_loss(train_y)

            net.layers[layer_idx].weights = pert_mat2
            net.forward(train_x, training=True)
            f2 = net.eval_loss(train_y)

            num_grad = (f1-f2)/(2*epsilon)
            net.layers[layer_idx].weights = backup_w_mat

            # In case of multi-head NN sum the changes relative to each weight
            if not np.isscalar(num_grad):
                # Remove empty dimensions
                num_grad = np.squeeze(num_grad)
                # Sum the effects of the different heads
                num_grad = np.sum(num_grad)

            unit_vec.append(num_grad)

        # Bias
        new_bias = np.array(backup_b_vec, copy=True)
        eye_vec = np.zeros((n_units, ))
        eye_vec[i] = 1

        eps_vec = epsilon*eye_vec
        pert_vec1 = new_bias + eps_vec
        pert_vec2 = new_bias - eps_vec

        net.layers[layer_idx].bias = pert_vec1
        net.forward(train_x, training=True)
        f1 = net.eval_loss(train_y)

        net.layers[layer_idx].bias = pert_vec2
        net.forward(train_x, training=True)
        f2 = net.eval_loss(train_y)

        num_grad = (f1-f2)/(2*epsilon)
        net.layers[layer_idx].bias = backup_b_vec

        # In case of multi-head NN sum the changes relative to each weight
        if not np.isscalar(num_grad):
            # Remove empty dimensions
            num_grad = np.squeeze(num_grad)
            # Sum the effects of the different heads
            num_grad = np.sum(num_grad)


        num_grad_b_vec.append(num_grad)
        num_grad_w_mat.append(unit_vec)

    num_patt = len(train_x)
    num_grad_w_mat = np.divide(num_grad_w_mat, num_patt)
    num_grad_b_vec = np.divide(num_grad_b_vec, num_patt)

    grad_w_diff = np.abs(an_grad_w-num_grad_w_mat)
    grad_b_diff = np.abs(an_grad_b-num_grad_b_vec)

    # Check if analytical and numerical gradient are close
    if (grad_w_diff > err_tol).any() or (grad_b_diff > err_tol).any():
        raise RuntimeError(f"Gradient precision is below threshold {err_tol}")

    if debug_bool:
        print(f"Layer {layer_idx}")
        print("Numerical grad: ")
        print(np.array(num_grad_w_mat))
        print("Analytical grad: \n", an_grad_w)
