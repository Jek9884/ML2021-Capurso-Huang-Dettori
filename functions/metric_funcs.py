import numpy as np
from functions.function import Function

# Hp: all outputs must be numpy.ndarray

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

    return np.array((tp+tn)/(tp+tn+fp+fn), ndmin=1)

def miscl_error(exp_mat, pred_mat):

    return np.array(1 - accuracy(exp_mat, pred_mat), ndmin=1)

def precision(exp_mat, pred_mat):

    tp, tn, fp, fn = compute_confusion_matrix_bin(exp_mat, pred_mat)

    return np.array(tp/(tp+fp), ndmin=1)

def recall(exp_mat, pred_mat):

    tp, tn, fp, fn = compute_confusion_matrix_bin(exp_mat, pred_mat)

    return np.array(tp/(tp+fn), ndmin=1)


# Metric function dictionary
acc_func = Function(accuracy, 'accuracy', 'max')
miscl_func = Function(miscl_error, 'miscl. error', 'min')
prec_func = Function(precision, 'precision', 'max')
rec_func = Function(recall, 'recall', 'max')

metr_dict = {
    'accuracy': acc_func,
    'miscl. error': miscl_func,
    'precision': prec_func,
    'recall': rec_func
}
