from src.feature_extraction.database_loader_new import *
from src.training.train_and_test import *
from src.internal_parameters import *
import argparse
import os
import logging
import pickle

def num_model_parameters(model):
    ps = []
    for name, p in model.named_parameters():
        ps.append(np.prod(p.size()))
    return sum(ps)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input query json file", type = str)
    parser.add_argument("--data", help="path to dataset directory", type = str)
    parser.add_argument("--dbname", help="dbname (imdb-small, imdb-medium, ...)", type = str)
    parser.add_argument("--model", help="directory path to save model files", type = str)
    parser.add_argument("--encode", help="generate encoding on the fly", action="store_true")
    
    
    parser.add_argument("--hid1", help="# of hidden units 1", type = int, default=128)
    parser.add_argument("--hid2", help="# of hidden units 2", type = int, default=256)
    parser.add_argument("--batch", help="training batch size", type = int, default=64)
    parser.add_argument("--lr", help="training learning rate", type = float, default=0.001)
    parser.add_argument("--epochs", help="training epochs", type = int, default=20)

    parser.add_argument("--num_train", help="number of training queries", type = int, default=100000)
    parser.add_argument("--num_valid", help="number of validation queries", type = int, default=10000)

    parser.add_argument("--gpu", help="id of gpu to use", type = int, default=0)

    args = parser.parse_args()

    
    input_path = args.input
    data_path = args.data
    dbname = args.dbname

    hid1 = args.hid1
    hid2 = args.hid2
    batch = args.batch
    lr = args.lr
    epochs = args.epochs


    timestamp = time.strftime("%Y%m%d-%H%M%S")

    log_path = "/dev/null"
    # os.makedirs('logs', exist_ok=True)
    # log_path = "logs/e2e_" + exp_name + "_" + timestamp + ".log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ])
    logger = logging.getLogger(__name__)
    logger.info("Input args: %r", args)

    minmax_path, wordvector_path, columns = prepare_loading(dbname)
    _, data=load_dataset(data_path)
    _, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, _ = prepare_dataset_general(data, dbname, columns)
    logger.info('data prepared')
    min_max_column = load_numeric_min_max_csv(minmax_path)
    logger.info('min_max loaded')

    index_total_num = len(indexes_id)
    table_total_num = len(tables_id)
    column_total_num = len(columns_id)
    physic_op_total_num = len(physic_ops_id)
    compare_ops_total_num = len(compare_ops_id)
    bool_ops_total_num = len(bool_ops_id)
    condition_op_dim = bool_ops_total_num + compare_ops_total_num+column_total_num+1000
    plan_node_max_num, condition_max_num, card_label_min, card_label_max = obtain_upper_bound_query_size_nocost(input_path)
    logger.info('query upper size prepared')

    if "job" in dbname or "imdb" in dbname:
        is_imdb=True
    else: 
        is_imdb=False

    cost_label_min, cost_label_max = 0.0, 1.0
    parameters = Parameters(condition_max_num, indexes_id, tables_id, columns_id, physic_ops_id, column_total_num,
                            table_total_num, index_total_num, physic_op_total_num, condition_op_dim, compare_ops_id, bool_ops_id,
                            bool_ops_total_num, compare_ops_total_num, data, min_max_column, wordvector_path, cost_label_min,
                            cost_label_max, card_label_min, card_label_max, hid1, hid2, batch, lr , is_imdb)
    
    num_train_batch = int(args.num_train / batch)
    num_valid_batch = int(args.num_valid / batch)
    
    
    max_queries = args.num_train + args.num_valid

    #encode input features online (for small database only)
    if args.encode:
        plan_batches = encode_plan_seq(input_path, parameters, batch, max_queries)

    model_dir_path = args.model
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)


    parameters_file_path = f'{model_dir_path}/parameters.pkl'
    parameters_file = open(parameters_file_path, "wb")
    pickle.dump(parameters, parameters_file)


    if (args.encode):
        model = train(0, num_train_batch, num_train_batch, num_train_batch + num_valid_batch, epochs, parameters, plan_batches, model_dir_path, args.gpu)
    else:
        encode_dir_path = input_path.replace(".json", "_enc")
        model = load_and_train(0, num_train_batch, num_train_batch, num_train_batch + num_valid_batch, epochs, parameters, encode_dir_path, model_dir_path, args.gpu)

    num_params = num_model_parameters(model)
    logger.info(f'# model parameters = {num_params}')
    logger.info(f'Size in MB = {num_params*4/1024/1024}')


if __name__ == "__main__":
    main()
