import numpy as np


def compute_confusion_matrix_bin(exp_vec, pred_vec):

    exp_vec = np.asarray(exp_vec)
    pred_vec = np.asarray(pred_vec)
    len_ex = exp_vec.shape[0]

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len_ex):

        if exp_vec[i] == pred_vec[i] and exp_vec[i] == 1:
            tp += 1
        elif exp_vec[i] == pred_vec[i] and exp_vec[i] == 0:
            tn += 1
        elif exp_vec[i] != pred_vec[i] and exp_vec[i] == 1:
            fn += 1
        elif exp_vec[i] != pred_vec[i] and exp_vec[i] == 0:
            fp += 1

    return tp, tn, fp, fn

def accuracy(exp_mat, pred_mat):

    tp, tn, fp, fn = compute_confusion_matrix_bin(exp_mat, pred_mat)

    return (tp+tn)/(tp+tn+fp+fn)

def miscl_error(exp_mat, pred_mat):

    return 1 - accuracy(exp_mat, pred_mat)

def precision(exp_mat, pred_mat):

    tp, tn, fp, fn = compute_confusion_matrix_bin(exp_mat, pred_mat)

    return tp/(tp+fp)

def recall(exp_mat, pred_mat):

    tp, tn, fp, fn = compute_confusion_matrix_bin(exp_mat, pred_mat)

    return tp/(tp+fn)
