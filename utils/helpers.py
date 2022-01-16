import os
import shutil
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
            ma_mask = [False] * len_shape[0] + [True] * len_mask
        else:
            elem_false = [False] * elem_size
            elem_true = [True] * elem_size
            ma_mask = [elem_false] * len_shape[0] + [elem_true] * len_mask

        new_ma_row = np.ma.array(new_row, mask=ma_mask)
        ma_array_list.append(new_ma_row)

    return np.ma.array(ma_array_list)


def geomspace_round(start, stop, num, decimals=6):

    geom_vals = np.geomspace(start, stop, num)
    return np.around(geom_vals, decimals)


def result_to_str(result, sep=' '):
    combo_str = ''

    for k, v in result['combo_net'].items():
        if k == "conf_layers":
            combo_str += ''.join(str(v)) + sep
        else:
            combo_str += str(v) + sep

    for value in result['combo_opt'].values():
        combo_str += str(value) + sep

    combo_str += str(result['score_tr']["avg"]) + sep
    combo_str += str(result['score_tr']["std"]) + sep
    combo_str += str(result['score_tr']["perc_25"]) + sep
    combo_str += str(result['score_tr']["perc_50"]) + sep
    combo_str += str(result['score_tr']["perc_75"]) + sep

    if result['score_val'] is not None:
        combo_str += str(result['score_val']["avg"]) + sep
        combo_str += str(result['score_val']["std"]) + sep
        combo_str += str(result['score_tr']["perc_25"]) + sep
        combo_str += str(result['score_tr']["perc_50"]) + sep
        combo_str += str(result['score_tr']["perc_75"]) + sep

    combo_str += str(result['epochs']) + sep
    combo_str += str(result['age']) + sep

    combo_str += '\n'

    return combo_str


def get_csv_header(result, sep=' '):
    header = ''

    for key in result['combo_net']:
        header += key + sep

    for key in result['combo_opt']:
        header += key + sep

    header += result['metric'] + '_tr_avg' + sep
    header += result['metric'] + '_tr_std' + sep
    header += result['metric'] + '_tr_perc25' + sep
    header += result['metric'] + '_tr_perc50' + sep
    header += result['metric'] + '_tr_perc75' + sep

    if result['score_val'] is not None:
        header += result['metric'] + '_val_avg' + sep
        header += result['metric'] + '_val_std' + sep
        header += result['metric'] + '_val_perc25' + sep
        header += result['metric'] + '_val_perc50' + sep
        header += result['metric'] + '_val_perc75' + sep

    header += 'epochs' + sep
    header += 'age' + sep
    header += '\n'

    return header


# Save all information related to the hyperparameter search results
def save_search_results(folder_path, results, search_net_dict,
                        search_opt_dict, csv_sep=';'):
    # If the folder doesn't exist, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    list_runs_str = os.listdir(folder_path)
    last_run = 1

    for run in list_runs_str:

        try:
            cur_run = int(run)

            if cur_run > last_run:
                last_run = cur_run

        except ValueError as ex:
            print(f"Invalid run name found in {folder_path}, {ex}")

    # Create a new run with next biggest integer as name
    last_run = str(last_run + 1)

    # Give a subfolder to each run
    folder_path = os.path.join(folder_path, last_run)
    os.makedirs(folder_path)

    file_path = os.path.join(folder_path, "results.csv")

    # Save results in results.csv
    with open(file_path, 'w', newline='') as file:

        header = get_csv_header(results[0], csv_sep)
        file.write(header)

        for i, res in enumerate(results):
            res_str = result_to_str(res, csv_sep)
            file.write(res_str)

            # Save plots if a plotter is available
            if res["plotter"] is not None:
                img_path = os.path.join(folder_path, f"{i + 1}.png")
                res["plotter"].savefig(img_path)


    # Save the search configuration used
    file_path = os.path.join(folder_path, "search.conf")

    with open(file_path, 'w', newline='') as file:
        file.write("Network search configuration:\n")

        for k, v in search_net_dict.items():
            file.write(str(k) + ": " + str(v) + "\n")

        file.write("\nOptimizer search configuration:\n")
        for k, v in search_opt_dict.items():
            file.write(str(k) + ": " + str(v) + "\n")


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

    if 'lr_decay_type' not in dict_opt:
        pass

    elif "lin" in dict_opt['lr_decay_type']:
        n_decay_tau = len(dict_opt['lr_dec_lin_tau'])
        np_combos = delete_combos(np_combos, combos_keys.index('lr_decay_type'), False,
                                  (combos_keys.index('lr_dec_lin_tau'), n_decay_tau))
        # print("lr_decay ", len(np_combos))

    elif "exp" in dict_opt['lr_decay_type']:
        n_decay_k = len(dict_opt['lr_dec_exp_k'])
        np_combos = delete_combos(np_combos, combos_keys.index('lr_decay_type'), False,
                                  (combos_keys.index('lr_dec_exp_k'), n_decay_k))

    elif None in dict_opt['lr_decay_type']:
        n_decay_tau = len(dict_opt['lr_dec_lin_tau'])
        n_decay_k = len(dict_opt['lr_dec_exp_k'])
        np_combos = delete_combos(np_combos, combos_keys.index('lr_decay_type'), False,
                                  (combos_keys.index('lr_dec_lin_tau'), n_decay_tau),
                                  (combos_keys.index('lr_dec_exp_k'), n_decay_k))


    if 'stop_crit_type' in dict_opt and 'fixed' in dict_opt['stop_crit_type']:
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
