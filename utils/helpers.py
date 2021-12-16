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
        combo_str += str(result['score_val'][0]) + sep  # average
        combo_str += str(result['score_val'][1])  # standard deviation

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


def clean_combos(dict_net, dict_opt, combos):
    net_keys, _ = zip(*dict_net.items())
    opt_keys, _ = zip(*dict_opt.items())

    combos_keys = net_keys + opt_keys

    np_combos = np.array([np.array(tup, dtype=object) for tup in combos])

    if 0 in dict_opt['momentum_val']:
        n_nesterov = len(dict_opt['nesterov'])
        np_combos = delete_combos(np_combos, combos_keys.index('momentum_val'), 0,
                                  (combos_keys.index('nesterov'), n_nesterov))
        # print("momentum ", len(np_combos))

    if "lin" in dict_opt['lr_decay_type']:
        n_decay_tau = len(dict_opt['lr_dec_lin_tau'])
        np_combos = delete_combos(np_combos, combos_keys.index('lr_decay_type'), False,
                                  (combos_keys.index('lr_dec_lin_tau'), n_decay_tau))
        # print("lr_decay ", len(np_combos))

    elif "exp" in dict_opt['lr_decay_type']:
        n_decay_k = len(dict_opt['lr_dec_exp_k'])
        np_combos = delete_combos(np_combos, combos_keys.index('lr_decay_type'), False,
                                  (combos_keys.index('lr_dec_exp_k'), n_decay_k))

    if 'fixed' in dict_opt['stop_crit_type']:
        n_patient = len(dict_opt['patient'])
        n_epsilon = len(dict_opt['epsilon'])
        np_combos = delete_combos(np_combos, combos_keys.index('stop_crit_type'), 'fixed',
                                  (combos_keys.index('epsilon'), n_epsilon),
                                  (combos_keys.index('patient'), n_patient))
        # print("stop_crit_type ", len(np_combos))

    return np_combos


def delete_combos(combos, target_idx, target_val, *dupl_fields_idx):

    # create an array of combos that are not affected by the cleaning process
    selected_idx = np.argwhere(combos[:, target_idx] != target_val)
    selected_idx = np.reshape(selected_idx, (len(selected_idx),))
    selected_combos = [*combos[selected_idx]]

    # copy the combos to clean
    combos_to_clean = combos[combos[:, target_idx] == target_val]

    # for each redundant field sort and keep only one of the possible values
    for field in dupl_fields_idx:
        combos_to_clean = combos_to_clean[np.lexsort((combos_to_clean[:, field[0]],))]
        combos_to_clean = combos_to_clean[:len(combos_to_clean) // field[1]]

    for combo in combos_to_clean:
        selected_combos.append(combo)

    return np.array(selected_combos)
