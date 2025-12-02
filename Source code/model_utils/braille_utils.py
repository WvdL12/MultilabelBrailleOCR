import numpy as np

MC_SIZE = 2 ** 6

def num_to_bin_string(num_label):
    return bin(int(num_label))[2:].zfill(6)[::-1]

def bin_to_string(bin_labels):
    if not isinstance(bin_labels, np.ndarray):
        return ''.join(bin_labels)
    return np.array([''.join(row.astype(str)) for row in bin_labels])

def string_to_bin(str_labels):
    return np.array([list(map(int, list(s))) for s in str_labels])

def num_to_bin(num_labels, w=6):
    if not isinstance(num_labels, np.ndarray):
        num_labels = np.array(num_labels)
    powers = 2 ** np.arange(w)
    return ((num_labels.reshape(-1,1) & powers) > 0).astype(int)

def bin_to_num(bin_labels):
    if not isinstance(bin_labels, np.ndarray):
        bin_labels = np.array(bin_labels)
    shape = bin_labels.shape[bin_labels.ndim > 1]

    powers = 2 ** np.arange(shape)
    return np.dot(bin_labels, powers)

def num_to_cat(num_labels):
    cat_matrix = np.zeros((num_labels.size, MC_SIZE), dtype=int)
    cat_matrix[np.arange(num_labels.size), num_labels] = 1
    
    return cat_matrix

def cat_to_num(cat_labels):
    return np.argmax(cat_labels, axis=1)
    
def ml_to_mc(data_y):
    return num_to_cat(bin_to_num(data_y))

def mc_to_ml(data_y):
    return num_to_bin(cat_to_num(data_y))