import matplotlib.pyplot as plt
import numpy as np

# TODO: lookup gradient checking technique

def plot_learning_curve(network, optimizer, train_x, train_y, tot_epochs, inter_epochs, metric):

    results_tr = []

    for i in range(tot_epochs//inter_epochs):

        optimizer.train(network, train_x, train_y, epochs=inter_epochs)

        if metric.name == "nll":
            pred_vec = network.forward(train_x, net_out=True)
        elif metric.name == "squared":
            pred_vec = network.forward(train_x)
        else:
            pred_vec = network.forward(train_x)
            pred_vec[pred_vec < 0.5] = 0
            pred_vec[pred_vec >= 0.5] = 1

        metric_res = metric(train_y, pred_vec)

        results_tr.append(metric_res)

    results_tr = np.array(results_tr)
    plt.plot(range(0, tot_epochs, inter_epochs), results_tr)
    plt.xlabel("Epochs")
    plt.ylabel(f"Metric ({metric.name})")
    plt.show()


# TODO: decide how to handle different norms for bias and weights
def plot_gradient_norm(network, optimizer, train_x, train_y, tot_epochs, inter_epochs):

    net_size = len(network.layers)
    fig, axs = plt.subplots(net_size, 2)

    results_w = {i: [] for i in range(net_size)}
    results_b = {i: [] for i in range(net_size)}

    for i in range(tot_epochs//inter_epochs):

        optimizer.train(network, train_x, train_y, epochs=inter_epochs)

        for j, layer in enumerate(network.layers):

            norm_grad_w = np.linalg.norm(layer.grad_w)
            norm_grad_b = np.linalg.norm(layer.grad_b)

            results_w[j].append(norm_grad_w)
            results_b[j].append(norm_grad_b)

    for i in range(net_size):
        axs[i][0].plot(range(0, tot_epochs, inter_epochs), results_w[i])
        axs[i][0].set_title(f"grad w (Layer {i})")
        axs[i][0].set_xlabel("Epochs")
        axs[i][0].set_ylabel("Norm value")

        axs[i][1].plot(range(0, tot_epochs, inter_epochs), results_b[i],
                       label=f"grad b (Layer {i})")
        axs[i][1].set_title(f"grad b (Layer {i})")
        axs[i][1].set_ylabel("Norm value")
        axs[i][1].set_xlabel("Epochs")

    plt.show()
