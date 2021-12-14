import numpy as np


def average_non_std_mat(value_mat):
    max_col = 0

    # Find longest row of non-standard matrix
    for row in value_mat:
        len_row = len(row)

        if len_row > max_col:
            max_col = len_row

    # Needed to handle multi-out network, in which each cell in value_mat
    # contains an array of len equal to the number of outs
    # Hp: all cells have the same size
    if np.isscalar(value_mat[0][0]):
        len_cell = 1
    else:
        len_cell = value_mat[0][0].shape[0]

    # Numerator of average
    sum_vec = np.zeros((max_col, len_cell))
    # Denominator of average
    count_vec = np.zeros((max_col, len_cell))
    # The array are squeezed instead of using an if to handle len_cell cases

    for row in value_mat:

        # In case of empty row
        if row == []:
            continue

        len_row = len(row)
        new_vec = np.array(row)
        loc_count_vec = np.ones((len_row, len_cell))

        # In case the matrix contains scalars reshape them in to array of len 1
        if np.isscalar(value_mat[0][0]):
            new_vec = np.reshape(new_vec, (len_row, 1))

        # Create new row with normalised length
        for i in range(max_col - len_row):
            vec_zeroes = np.zeros((1, len_cell))
            new_vec = np.append(new_vec, vec_zeroes, axis=0)
            loc_count_vec = np.append(loc_count_vec, vec_zeroes, axis=0)

        sum_vec += new_vec
        count_vec += loc_count_vec

    div_vec = np.divide(sum_vec, count_vec)

    # Needed in order to avoid sending a matrix of values as count
    if len_cell > 1:
        count_vec = count_vec[:, 0]

    return div_vec, count_vec


def result_to_str(result, sep=' '):
    combo_str = str(result['combo_net']['conf_layers']) + sep

    combo_str += result['combo_net']['init_func'].name + sep
    combo_str += result['combo_net']['act_func'].name + sep
    combo_str += result['combo_net']['out_func'].name + sep
    combo_str += result['combo_net']['loss_func'].name + sep

    for value in result['combo_opt'].values():
        combo_str += str(value) + sep

    combo_str += str(result['score_tr'][0]) + sep  # average
    combo_str += str(result['score_tr'][1])  # standard deviation

    if result['score_val'] is not None:
        combo_str += sep
        combo_str += str(result['score_val'][0]) + sep # average
        combo_str += str(result['score_val'][1]) # standard deviation

    combo_str += '\n'

    return combo_str


def get_csv_header(result, sep=' '):
    header = ''

    for key in result['combo_net']:
        header += key + sep

    for key in result['combo_opt']:
        header += key + sep

    header += result['metric'] + '_tr_avg' + sep
    header += result['metric'] + '_tr_std'

    if result['score_val'] is not None:
        header += sep
        header += result['metric'] + '_tr_avg' + sep
        header += result['metric'] + '_tr_std'

    header += '\n'

    return header


def save_results_to_csv(path, results, sep=';'):
    with open(path, 'w') as file:
        file.write(get_csv_header(results[0], sep))
        for combo in results:
            file.write(result_to_str(combo, sep))
