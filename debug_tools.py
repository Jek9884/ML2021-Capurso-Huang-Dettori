import matplotlib.pyplot as plt
import numpy as np

# TODO: lookup gradient checking technique

def plot_learning_curve(network, optimizer, train_x, train_y, tot_epochs, inter_epochs, metric):

    results_tr = []

    for i in range(tot_epochs//inter_epochs):

        optimizer.train(network, train_x, train_y, epochs=inter_epochs)

        pred_vec = network.forward(train_x)
        pred_vec[pred_vec < 0.5] = 0
        pred_vec[pred_vec >= 0.5] = 1

        results_tr.append(metric(train_y, pred_vec))

    plt.plot(range(0, tot_epochs, inter_epochs), results_tr)
    plt.xlabel("Epochs")
    plt.ylabel("Metric")
    plt.show()
