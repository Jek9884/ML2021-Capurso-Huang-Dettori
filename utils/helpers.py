import numpy as np

def average_non_std_mat(value_mat):

    max_col = 0

    # Find longest row of non-standard matrix
    for row in value_mat:
        len_row = len(row)

        if len_row > max_col:
            max_col = len_row

    len_cell = 1
    # Needed to handle multi-out network, in which each cell in value_mat
    # contains an array of len equal to the number of outs
    # Hp: all cells have the same size
    if isinstance(value_mat[0][0], np.ndarray):
        len_cell = value_mat[0][0].shape[0]

    # Numerator of average
    sum_vec = np.squeeze(np.zeros((max_col, len_cell)))
    # Denominator of average
    count_vec = np.squeeze(np.zeros((max_col, len_cell)))
    # The array are squeezed instead of using an if to handle len_cell cases

    for row in value_mat:

        len_row = len(row)
        new_vec = np.array(row)
        loc_count_vec = np.squeeze(np.ones((len_row, len_cell)))

        # Create new row with normalised length
        for i in range(max_col - len_row):

            if len_cell == 1:
                vec_zeroes = np.zeros((len_cell,))
            else:
                vec_zeroes = np.zeros((1, len_cell))

            new_vec = np.append(new_vec, vec_zeroes, axis=0)
            loc_count_vec = np.append(loc_count_vec, vec_zeroes, axis=0)

        sum_vec += new_vec
        count_vec += loc_count_vec

    div_vec = np.divide(sum_vec, count_vec)

    return div_vec, count_vec
