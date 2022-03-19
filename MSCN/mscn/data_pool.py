import csv
import torch
from pathlib import Path
from torch.utils.data import dataset

from mscn.util_common import *
from mscn.util_pool import *
from mscn.data_common import *


def load_and_encode_train_data_from_path(query_file_path, dbname, num_train, num_valid, use_string = False):

    sep, string_columns, minmax_file_path, sample_file_path, word_vectors_path, alias_dict = prepare_loading(query_file_path, dbname)
    

    print("query_file_path = " + query_file_path)
    print("sample_file_path = " + sample_file_path)
    print("minmax_file_path = " + minmax_file_path)
    if(use_string):
        print("wordvector_file_path = " + word_vectors_path)
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
    if (use_string):
        is_imdb = ("job" in dbname) or ("imdb" in dbname)
        features_enc, masks, feature_sizes = encode_data(tables, samples, predicates, joins, column_min_max_vals, table2vec, column2vec, op2vec, join2vec, string_columns, word_vectors_path, is_imdb)
    else:
        features_enc, masks, feature_sizes = encode_data(tables, samples, predicates, joins, column_min_max_vals, table2vec, column2vec, op2vec, join2vec)
    label_norm, min_val, max_val = normalize_labels(label)

    # Split in training and validation samples
    assert num_train <= num_queries
    features_train = features_enc[:num_train]
    masks_train = masks[:num_train]
    labels_train = label_norm[:num_train]

    features_valid = features_enc[num_train:num_train + num_valid]
    masks_valid = masks[num_train:num_train + num_valid]
    labels_valid = label_norm[num_train:num_train + num_valid]

    print("Number of training samples: {}".format(len(labels_train)))
    print("Number of validation samples: {}".format(len(labels_valid)))

    dicts = [table2vec, column2vec, op2vec, join2vec]
    train_data = [features_train, masks_train]
    validation_data = [features_valid, masks_valid]
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_valid, feature_sizes, train_data, validation_data


def make_dataset(features, masks, labels):
    """Add zero-padding and wrap as tensor dataset."""

    feature_tensors = torch.FloatTensor(features)

    mask_tensors = torch.FloatTensor(masks)

    target_tensor = torch.FloatTensor(labels)

    return dataset.TensorDataset(feature_tensors, mask_tensors, target_tensor)


def get_train_dataset_from_path(train_file_path, dbname, num_train, num_valid, use_string):
    dicts, column_min_max_vals, min_val, max_val, labels_train, labels_valid, feature_sizes, train_data, validation_data = load_and_encode_train_data_from_path(
        train_file_path, dbname, num_train, num_valid, use_string)
    train_dataset = make_dataset(*train_data, labels=labels_train)
    validation_dataset = make_dataset(*validation_data, labels=labels_valid)
        
    print("Created TensorDataset for training data")
    return dicts, column_min_max_vals, min_val, max_val, labels_train, labels_valid, train_dataset, validation_dataset, feature_sizes

