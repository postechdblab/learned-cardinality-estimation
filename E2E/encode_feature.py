from src.feature_extraction.database_loader_new import *
from src.training.train_and_test import *
from src.internal_parameters import *
import argparse
import os
import logging
import pickle


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input query json file", type = str)
    parser.add_argument("--data", help="path to sample directory", type = str)
    parser.add_argument("--dbname", help="dbname (schema)", type = str)
    parser.add_argument("--batch", help="training batch size", type = int, default=64)
    parser.add_argument("--num", help="number of queries to encode", type = int, default=110000)

    args = parser.parse_args()

    input_path = args.input
    data_path = args.data
    dbname = args.dbname

    batch = args.batch

    minmax_path, wordvector_path, columns = prepare_loading(dbname)
    _, data=load_dataset(data_path)
    _, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, _ = prepare_dataset_general(data, dbname, columns)
    print('data prepared')
    min_max_column = load_numeric_min_max_csv(minmax_path)
    print('min_max loaded')

    index_total_num = len(indexes_id)
    table_total_num = len(tables_id)
    column_total_num = len(columns_id)
    physic_op_total_num = len(physic_ops_id)
    compare_ops_total_num = len(compare_ops_id)
    bool_ops_total_num = len(bool_ops_id)
    condition_op_dim = bool_ops_total_num + compare_ops_total_num+column_total_num+1000
    _, condition_max_num, card_label_min, card_label_max = obtain_upper_bound_query_size_nocost(input_path)
    print('query upper size prepared')

    if "job" in dbname or "imdb" in dbname:
        is_imdb=True
    else: 
        is_imdb=False

    # we do not use cost labels
    cost_label_min, cost_label_max = 0.0, 1.0
    parameters = Parameters(condition_max_num, indexes_id, tables_id, columns_id, physic_ops_id, column_total_num,
                            table_total_num, index_total_num, physic_op_total_num, condition_op_dim, compare_ops_id, bool_ops_id,
                            bool_ops_total_num, compare_ops_total_num, data, min_max_column, wordvector_path, cost_label_min,
                            cost_label_max, card_label_min, card_label_max, None, None, batch, None ,None, is_imdb)

    encode_dir_path = input_path.replace(".json", "_enc")
    if not os.path.exists(encode_dir_path):
        os.makedirs(encode_dir_path, exist_ok=True)
        
    encode_train_plan_seq_save(input_path, parameters, batch, encode_dir_path, args.num)


if __name__ == "__main__":
    main()
