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


def plot_gradient_norm(network, optimizer, train_x, train_y, tot_epochs, inter_epochs, norm="fro"):

    for i in range(tot_epochs//inter_epochs):

        optimizer.train(network, train_x, train_y, epochs=inter_epochs)

        for j, layer in enumerate(network.layers):

            norm_grad_w = np.linalg.norm(layer.grad_w, ord=norm)
            norm_grad_b = np.linalg.norm(layer.grad_w, ord=norm)

            plt.scatter(i*inter_epochs, norm_grad_w, label=f"grad_w ({j})")
            plt.scatter(i*inter_epochs, norm_grad_b, label=f"grad_b ({j})")

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(f"Norm value ({norm})")
    plt.show()
