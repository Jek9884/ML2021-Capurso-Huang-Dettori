import numpy as np


def convert_ragged_mat_to_ma_array(ragged_mat):

    ma_array_list = []

    # First: find longest row of non-standard matrix
    max_col = 0

    for row in ragged_mat:
        len_row = len(row)

        if len_row > max_col:
            max_col = len_row

    for row in ragged_mat:

        if row == []:
            continue

        len_shape = np.shape(row)
        len_mask = max_col - len_shape[0]

        if np.isscalar(row[0]):
            elem_size = 0
            new_row = np.empty((len_mask,))
        else:
            elem_size = len(row[0])
            new_row = np.empty((len_mask, len_shape[1]))

        new_row = np.insert(new_row, 0, row, axis=0)

        if elem_size == 0:
            ma_mask = [False]*len_shape[0] + [True]*len_mask
        else:
            elem_false = [False]*elem_size
            elem_true = [True]*elem_size
            ma_mask = [elem_false]*len_shape[0] + [elem_true]*len_mask

        new_ma_row = np.ma.array(new_row, mask=ma_mask)
        ma_array_list.append(new_ma_row)

    return np.ma.array(ma_array_list)


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
