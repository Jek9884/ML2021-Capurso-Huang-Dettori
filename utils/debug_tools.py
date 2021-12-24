import numpy as np

from network import Network

# Implementation of gradient checker

def check_gradient_net(par_combo_net, train_x, train_y):

    net = Network(**par_combo_net)

    # Note: supports only "online learning"
    for i in range(train_x.shape[0]):

        x_row = np.array([train_x[i]])
        y_row = np.array([train_y[i]])

        for j, _ in enumerate(net.layers):

            if j == len(net.layers) - 1:
                check_gradient_layer(net, j, x_row, y_row)
            else:
                check_gradient_layer(net, j, x_row, y_row)

def check_gradient_layer(net, layer_idx, train_x, train_y, epsilon=10**-6):

    x_row = train_x
    y_row = train_y
    weights_mat = net.layers[layer_idx].weights
    n_units = weights_mat.shape[0]
    n_feat = weights_mat.shape[1]

    num_grad_mat = []

    # Perturbe wrt all the weights
    for i in range(n_units):

        unit_vec = []
        for j in range(n_feat):

            new_weights = np.array(weights_mat, copy=True)
            eye_mat = np.zeros((n_units, n_feat))
            eye_mat[i, j] = 1

            eps_mat = epsilon*eye_mat
            pert_mat1 = new_weights + eps_mat
            pert_mat2 = new_weights - eps_mat

            net.layers[layer_idx].weights = pert_mat1
            net.forward(x_row, training=True)
            f1 = net.eval_loss(y_row)

            net.layers[layer_idx].weights = pert_mat2
            net.forward(x_row, training=True)
            f2 = net.eval_loss(y_row)

            num_grad = np.squeeze((f1-f2)/(2*epsilon), axis=0)
            net.layers[layer_idx].weights = weights_mat

            # In case of multi-head NN sum the changes relative to each weight
            if not np.isscalar(num_grad) and len(num_grad) > 1:
                num_grad = np.sum(num_grad)

            unit_vec.append(num_grad)

        num_grad_mat.append(unit_vec)

    print(f"Layer {layer_idx}")
    print("Numerical grad: ")
    print(np.array(num_grad_mat))

    # Analytical gradient
    net.null_grad()
    net.forward(x_row, training=True)
    net.backward(y_row)
    an_grad = net.layers[layer_idx].grad_w

    print("Analytical grad: \n", an_grad)
