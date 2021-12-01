import matplotlib.pyplot as plt
import numpy as np

# TODO: lookup gradient checking technique

def plot_learning_curve(network, optimizer, train_x, train_y, tot_epochs, inter_epochs, metric):

    results_tr = []

    for i in range(tot_epochs//inter_epochs):

        optimizer.train(network, train_x, train_y, epochs=inter_epochs)

        if metric.name == "nll":
            out_vec = network.forward(train_x, net_out=True)
            metric_res = metric(train_y, out_vec)
            metric_res = np.sum(metric_res)
        else:
            pred_vec = network.forward(train_x)
            pred_vec[pred_vec < 0.5] = 0
            pred_vec[pred_vec >= 0.5] = 1

            metric_res = metric(train_y, pred_vec)

        results_tr.append(metric_res)

    plt.plot(range(0, tot_epochs, inter_epochs), results_tr)
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.show()
