from mscn.data_common import *
from mscn.util_common import *
from mscn.util_fcn import *


def get_fcn_variable_pred_enc_size(column2vec, op2vec, dbname):
    if ("job" in dbname) or ("imdb" in dbname):
        string_columns = IMDB_STRING_COLUMNS
    elif "tpcds" in dbname:
        string_columns = TPCDS_STRING_COLUMNS
    else:
        raise
    pred_enc_size = 0
    for column in column2vec:
        pred_enc_size += len(op2vec)
        if column in string_columns:
            pred_enc_size += STR_EMB_SIZE
        else:
            pred_enc_size += 1
    
    return pred_enc_size

    
def load_and_encode_train_data_from_path(query_file_path, dbname, num_train, num_valid, use_string, var):

    sep, string_columns, minmax_file_path, sample_file_path, word_vectors_path, alias_dict = prepare_loading(query_file_path, dbname)
    print("query_file_path = " + query_file_path)
    print("sample_file_path = " + sample_file_path)
    joins, predicates, tables, samples, label = load_data_from_path(query_file_path, sample_file_path, sep)
    num_queries = len(tables)

    # Get column name dict
    column_names = get_all_column_names(predicates)
    column2vec, idx2column = get_set_encoding(column_names)

    # Get table name dict
    table_names = get_all_table_names(tables)
    table2vec, idx2table = get_set_encoding(table_names)

    # Get operator name dict
    operators = get_all_operators(predicates)
    op2vec, idx2op = get_set_encoding(operators)

    # Get join name dict
    join_set = get_all_joins(joins)
    join2vec, idx2join = get_set_encoding(join_set)

    # Get min and max values for each column
    with open(minmax_file_path, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        column_min_max_vals = {}
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

    # Get feature encoding and proper normalization
    # samples_enc = encode_samples(tables, samples, table2vec)
    tables_enc, samples_enc = encode_samples(tables, samples, table2vec)
    if use_string:
        is_imdb = ("job" in dbname) or ("imdb" in dbname)
        if var:
            predicates_enc, joins_enc = encode_data_with_string_var(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec, string_columns, word_vectors_path, is_imdb)
        else:
            predicates_enc, joins_enc = encode_data_with_string(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec, string_columns, word_vectors_path, is_imdb)
    else:
        predicates_enc, joins_enc = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    label_norm, min_val, max_val = normalize_labels(label)

    # Split in training and validation samples
    assert num_train <= num_queries

    tables_train = tables_enc[:num_train]
    samples_train = samples_enc[:num_train]
    predicates_train = predicates_enc[:num_train]
    joins_train = joins_enc[:num_train]
    labels_train = label_norm[:num_train]

    tables_valid = tables_enc[num_train:num_train + num_valid]
    samples_valid = samples_enc[num_train:num_train + num_valid]
    predicates_valid = predicates_enc[num_train:num_train + num_valid]
    joins_valid = joins_enc[num_train:num_train + num_valid]
    labels_valid = label_norm[num_train:num_train + num_valid]

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_valid)))


    dicts = [table2vec, column2vec, op2vec, join2vec]
    train_data = [tables_train, samples_train, predicates_train, joins_train]
    validation_data = [tables_valid, samples_valid, predicates_valid, joins_valid]
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_valid, train_data, validation_data


def make_dataset(tables, samples, predicates, joins, labels):
    """Add zero-padding and wrap as tensor dataset."""

    # sample_masks = []

    table_tensors = np.vstack(tables)
    table_tensors = torch.FloatTensor(table_tensors)

    sample_tensors = np.vstack(samples)
    sample_tensors = torch.FloatTensor(sample_tensors)
    
    predicate_tensors = np.vstack(predicates)
    predicate_tensors = torch.FloatTensor(predicate_tensors)

    join_tensors = np.vstack(joins)
    join_tensors = torch.FloatTensor(join_tensors)

    target_tensor = torch.FloatTensor(labels)
    return dataset.TensorDataset(table_tensors, sample_tensors, predicate_tensors, join_tensors, target_tensor)


def get_train_dataset_from_path(train_file_path, dbname, num_train, num_valid, use_string, var = False):
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_valid, train_data, validation_data = load_and_encode_train_data_from_path(
        train_file_path, dbname, num_train, num_valid, use_string, var)
    train_dataset = make_dataset(*train_data, labels=labels_train)
    validation_dataset = make_dataset(*validation_data, labels=labels_valid)
    print("Created TensorDataset for training data")
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_valid,  train_dataset, validation_dataset


