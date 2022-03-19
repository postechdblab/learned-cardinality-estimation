import numpy as np
import re
import hashlib
from operator import add
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from datetime import datetime
import torch

from mscn.canonical import *

NUM_MATERIALIZED_SAMPLES = 1000
STR_EMB_SIZE = 1000
TPCDS_MIN_DATE="1900-01-02"
TPCDS_MAX_DATE="2100-01-01"

DATE_COLUMNS = {
    'dv.dv_create_date', 'd.d_date', 'i.i_rec_start_date', 'i.i_rec_end_date', 's.s_rec_start_date', 's.s_rec_end_date', 'cc.cc_rec_start_date', 'cc.cc_rec_end_date', 'web.web_rec_start_date' , 'web.web_rec_end_date', 'wp.wp_rec_start_date', 'wp.wp_rec_end_date'
}


def unnormalize_torch(vals, min_val, max_val):
    vals = (vals * (max_val - min_val)) + min_val
    return torch.exp(vals)


def qerror_loss(preds, targets, min_val, max_val):
    qerror = []
    preds = unnormalize_torch(preds, min_val, max_val)
    targets = unnormalize_torch(targets, min_val, max_val)

    for i in range(len(targets)):
        if preds[i] < 1: preds[i] = 1 #ceiling
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror))


def log_qerror(preds_unnorm, labels_unnorm,logger):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] < 1: preds_unnorm[i] = 1 #ceiling
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))
    
    logger.info("Mean: {}".format(np.mean(qerror)))
    logger.info("Min: {}".format(np.min(qerror)))
    logger.info("Median: {}".format(np.median(qerror)))
    logger.info("90th percentile: {}".format(np.percentile(qerror, 90)))
    logger.info("95th percentile: {}".format(np.percentile(qerror, 95)))
    logger.info("99th percentile: {}".format(np.percentile(qerror, 99)))
    logger.info("Max: {}".format(np.max(qerror)))
    return qerror


def mean_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] < 1: preds_unnorm[i] = 1 #ceiling
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))
    return np.mean(qerror)


# For string embedding
def determine_prefix(column, is_imdb = True):
    relation_name = column.split('.')[0]
    column_name = column.split('.')[1]
    if is_imdb:
        if relation_name == 'at':
            if column_name == 'title':
                return 'title_'
            elif column_name == 'imdb_index':
                return 'imdb_index_'
            elif column_name == 'phonetic_code':
                return 'phonetic_code_'
            elif column_name == 'note':
                return 'note_'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'ch_n':
            if column_name == 'name':
                return 'name_'
            elif column_name == 'name_pcode_nf':
                return 'nf_'
            elif column_name == 'surname_pcode':
                return 'surname_'
            elif column_name == 'imdb_index':
                return 'imdb_index_'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'mi_idx':
            if column_name == 'info':
                return 'info_'
            elif column_name == 'note':
                return 'note_'
            else:
                print (column)
                raise
        elif relation_name == 't':
            if column_name == 'title':
                return 'title_'
            elif column_name == 'imdb_index':
                return 'imdb_index_'
            elif column_name == 'phonetic_code':
                return 'phonetic_code_'
            elif column_name == 'series_years':
                return 'series_years'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'rt':
            if column_name == 'role':
                return 'role_'
            else:
                print (column)
                raise
        elif relation_name == 'mc':
            if column_name == 'note':
                return 'note_'
            else:
                print (column)
                raise
        elif relation_name == 'it':
            if column_name == 'info':
                return 'info_'
            else:
                print (column)
                raise
        elif relation_name == 'ct':
            if column_name == 'kind':
                return ''
            else:
                print (column)
                raise
        elif relation_name == 'cn':
            if column_name == 'name':
                return 'cn_name_'
            elif column_name == 'country_code':
                return 'country_'
            elif column_name == 'name_pcode_sf':
                return 'sf_'
            elif column_name == 'name_pcode_nf':
                return 'nf_'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'k':
            if column_name == 'keyword':
                return 'keyword_'
            elif column_name == 'phonetic_code':
                return 'phonetic_code_'
            else:
                print (column)
                raise
                
        elif relation_name == 'mi':
            if column_name == 'info':
                return ''
            elif column_name == 'note':
                return 'note_'
            else:
                print (column)
                raise
        elif relation_name == 'n':
            if column_name == 'gender':
                return 'gender_'
            elif column_name == 'name':
                return 'name_'
            elif column_name == 'name_pcode_cf':
                return 'cf_'
            elif column_name == 'name_pcode_nf':
                return 'nf_'
            elif column_name == 'surname_pcode':
                return 'surname_'
            elif column_name == 'imdb_index':
                return 'imdb_index_'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'an':
            if column_name == 'name':
                return 'name_'
            elif column_name == 'name_pcode_cf':
                return 'cf_'
            elif column_name == 'name_pcode_nf':
                return 'nf_'
            elif column_name == 'surname_pcode':
                return 'surname_'
            elif column_name == 'imdb_index':
                return 'imdb_index_'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'lt':
            if column_name == 'link':
                return 'link_'
            else:
                print (column)
                raise
        elif relation_name == 'pi':
            if column_name == 'note':
                return 'note_'
            elif column_name == 'info':
                return ''
            else:
                print (column)
                raise
        elif relation_name == 'ci':
            if column_name == 'note':
                return 'note_'
            else:
                print (column)
                raise
        elif relation_name == 'cct':
            if column_name == 'kind':
                return 'kind_'
            else:
                print (column)
                raise
        elif relation_name == 'kt':
            if column_name == 'kind':
                return 'kind_'
            else:
                print (column)
                raise
        else:
            print (relation_name)
            raise
    else:
        relation_name = column.split('.')[0]
        column_name = column.split('.')[1]
        # remove the table_name alias in tpcds column names
        prefix = column_name[column_name.find("_") + 1 : ]
        return prefix


def md5_hash_string_to_int(t):
    return int(hashlib.md5(t.encode()).hexdigest(), 16)


def get_string_embedding(word_vectors, column, value, is_imdb):
    prefix = determine_prefix(column, is_imdb)
    value = prefix + value    
    if value in word_vectors:
        embedded_result = np.array(list(word_vectors[value]), dtype=np.float32)
    else:
        embedded_result = np.zeros(int(STR_EMB_SIZE/2), dtype=np.float32)
    hash_result = np.zeros(int(STR_EMB_SIZE/2), dtype=np.float32)
    for t in value:
        hash_result[md5_hash_string_to_int(t) % 500] = 1.0
    return np.concatenate((embedded_result, hash_result), 0)


# Helper functions for data processing
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_all_column_names(predicates):
    column_names = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                column_names.add(column_name)
    return column_names


def get_all_table_names(tables):
    table_names = set()
    for query in tables:
        for table in query:
            table_names.add(table)
    return table_names


def get_all_operators(predicates):
    operators = set()
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                operator = predicate[1]
                operators.add(operator)
    return operators


def get_all_joins(joins):
    join_set = set()
    for query in joins:
        for join in query:
            join_set.add(join)
    return join_set


def idx_to_onehot(idx, num_elements):
    onehot = np.zeros(num_elements, dtype=np.float32)
    onehot[idx] = 1.
    return onehot


def get_set_encoding(source_set, onehot=True):
    num_elements = len(source_set)
    source_list = list(source_set)
    # Sort list to avoid non-deterministic behavior
    source_list.sort()
    # Build map from s to i
    thing2idx = {s: i for i, s in enumerate(source_list)}
    # Build array (essentially a map from idx to s)
    idx2thing = [s for i, s in enumerate(source_list)]
    if onehot:
        thing2vec = {s: idx_to_onehot(i, num_elements) for i, s in enumerate(source_list)}
        return thing2vec, idx2thing
    return thing2idx, idx2thing


def get_min_max_vals(predicates, column_names):
    min_max_vals = {t: [float('inf'), float('-inf')] for t in column_names}
    for query in predicates:
        for predicate in query:
            if len(predicate) == 3:
                column_name = predicate[0]
                val = float(predicate[2])
                if val < min_max_vals[column_name][0]:
                    min_max_vals[column_name][0] = val
                if val > min_max_vals[column_name][1]:
                    min_max_vals[column_name][1] = val
    return min_max_vals


def normalize_data(val, column_name, column_min_max_vals, emb_size = 1):
    val_norm = 0.0
    if val != "None":
        min_val = column_min_max_vals[column_name][0]
        max_val = column_min_max_vals[column_name][1]
        val = float(val)
        if max_val > min_val:
            val_norm = (val - min_val) / (max_val - min_val)
    vec = np.array(val_norm, dtype=np.float32)
    if emb_size > 1:
        vec = np.hstack([vec, np.zeros(emb_size -1, dtype=np.float32)])
    return vec

def normalize_date(val, emb_size = 1):
    dmin = datetime.strptime(TPCDS_MIN_DATE, "%Y-%m-%d")
    dmax = datetime.strptime(TPCDS_MAX_DATE, "%Y-%m-%d")
    if val == None or val == "None":
        val_norm = 0
    else:
        d = datetime.strptime(val, "%Y-%m-%d")
        val_norm = float((d-dmin).days) / (dmax-dmin).days
    
    vec = np.array(val_norm, dtype=np.float32)
    if emb_size > 1:
        vec = np.hstack([vec, np.zeros(emb_size -1, dtype=np.float32)])
    return vec
    

def normalize_labels(labels, min_val=None, max_val=None):
    labels = np.array([np.log(float(l)) for l in labels], dtype=np.float32)
    if min_val is None:
        min_val = labels.min()
        print("min log(label): {}".format(min_val))
    if max_val is None:
        max_val = labels.max()
        print("max log(label): {}".format(max_val))
    labels_norm = (labels - min_val) / (max_val - min_val)
    # Threshold labels
    labels_norm = np.minimum(labels_norm, 1)
    labels_norm = np.maximum(labels_norm, 0)
    return labels_norm, min_val, max_val


def unnormalize_labels(labels_norm, min_val, max_val):
    labels_norm = np.array(labels_norm, dtype=np.float32)
    labels = (labels_norm * (max_val - min_val)) + min_val
    return np.array(np.round(np.exp(labels)), dtype=np.int64)


def transfer_to_canonical_form(schema_joins, joins):
    new_joins = []
    added_tables = []
    invalid = set()
    schema_join_tuples = join_to_tuple(schema_joins)
    print(schema_join_tuples)
    for i, query_joins in enumerate(joins):
        if query_joins[0] == '': 
            new_query_joins = ['']
            added_tables.append([])
        else:
            query_join_tuples = join_to_tuple(query_joins)
            # print(query_join_tuples)
            succ, new_query_join_tuples, added_query_tables = fit_joins_to_schema(schema_join_tuples, query_join_tuples)
            if succ: 
                new_query_joins = tuple_to_join(new_query_join_tuples)
                added_tables.append(added_query_tables)
            else:
                new_query_joins = ['']
                added_tables.append([])
                invalid.add(i)
        new_joins.append(new_query_joins)
    return new_joins, added_tables, invalid


def num_model_parameters(model):
    ps = []
    for name, p in model.named_parameters():
        ps.append(np.prod(p.size()))
    return sum(ps)


def convert_tables(tables, alias_dict):
    return [[table + " " + alias_dict[table] for table in q_tables] for q_tables in tables]


def convert_tables_reverse(tables, alias_dict):
    rev_alias_dict = {v:k for k,v in alias_dict.items()}
    return [[rev_alias_dict[alias] + " " + alias for alias in q_tables] for q_tables in tables]


def convert_join(join, alias_dict):
    if len(join) == 0 : return join
    left = join.split("=")[0]
    right = join.split("=")[1]
    left_table = left.split(".")[0]
    left_column = left.split(".")[1]
    right_table = right.split(".")[0]
    right_column = right.split(".")[1]
    return f'{alias_dict[left_table]}.{left_column}={alias_dict[right_table]}.{right_column}'


def convert_joins(joins, alias_dict):
    return [[convert_join(join,alias_dict) for join in q_joins] for q_joins in joins]


def convert_pred(pred, alias_dict):
    if len(pred) != 3 : return pred
    new_pred = pred
    table = pred[0].split(".")[0]
    column = pred[0].split(".")[1]
    new_pred[0] = alias_dict[table] + "." + column
    return new_pred


def convert_preds(preds, alias_dict):
    return [[convert_pred(pred,alias_dict) for pred in q_preds] for q_preds in preds]


# filter IN and NOT_IN predicates on numeric columns
def filter_unsupported_predicates(predicates, invalid, string_columns):
    for i, query_preds in enumerate(predicates):
        if i in invalid: continue
        for pred in query_preds:
            if len(pred) != 3: continue
            col = pred[0]
            op = pred[1]
            val = pred[2]

            if col not in string_columns and (op == "IN" or op == "NOT IN"):
                invalid.add(i)
                break
    return invalid