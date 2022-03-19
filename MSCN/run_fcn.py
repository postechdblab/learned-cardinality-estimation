import argparse
import time
import os
import logging
import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from mscn.util_common import *
from mscn.util_fcn import *
from mscn.data_common import *
from mscn.data_fcn import *
from mscn.model import *

import logging

import torch


def predict(model, data_loader, gpu_id):
    preds = []
    times = []
    t_total = 0.

    model = model.cuda(gpu_id)
    model.eval()
    for batch_idx, data_batch in enumerate(data_loader):
        tables, samples, predicates, joins, targets = data_batch
        tables, samples, predicates, joins, targets = tables.cuda(gpu_id), samples.cuda(gpu_id), predicates.cuda(gpu_id), joins.cuda(gpu_id), targets.cuda(gpu_id)
        tables, samples, predicates, joins, targets = Variable(tables), Variable(samples), Variable(predicates), Variable(joins), Variable(
            targets)

        t = time.time()
        outputs = model(tables, samples, predicates, joins)
        t_batch = time.time() - t
        t_total += t_batch


        for i in range(outputs.data.shape[0]):
            preds.append(outputs.data[i])
            times.append(t_batch)

    return preds, t_total, times


def train(train_data, validation_data, md_dict, labels_train, labels_valid, num_epochs, batch_size, hid_units, learning_rate, logger, gpu_id, use_string):
    
    dicts = md_dict["dicts"]
    min_val = md_dict["min_val"]
    max_val = md_dict["max_val"]
    
    # Load training and validation data
    table2vec, column2vec, op2vec, join2vec = dicts

    # Train model
    table_feat_size = len(table2vec)
    sample_feat_size =  NUM_MATERIALIZED_SAMPLES
    num_columns = len(column2vec)
    
    if use_string:
        operand_size = STR_EMB_SIZE
    else:
        operand_size = 1
    predicate_feat_size = len(op2vec) + operand_size
    join_feat_size = len(join2vec)

    model = FCN(table_feat_size, sample_feat_size, num_columns, predicate_feat_size, join_feat_size, hid_units)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.cuda(gpu_id)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    validation_data_loader = DataLoader(validation_data, batch_size=batch_size)

    valid_time = 0

    t = time.time()
    model.train()

    for epoch in range(num_epochs):
        loss_total = 0.

        for batch_idx, data_batch in enumerate(train_data_loader):
            tables, samples, predicates, joins, targets = data_batch
            tables, samples, predicates, joins, targets = tables.cuda(gpu_id), samples.cuda(gpu_id), predicates.cuda(gpu_id), joins.cuda(gpu_id), targets.cuda(gpu_id)
            tables, samples, predicates, joins, targets = Variable(tables), Variable(samples), Variable(predicates), Variable(joins), Variable( targets)

            optimizer.zero_grad()
            outputs = model(tables, samples, predicates, joins)
            loss = qerror_loss(outputs, targets.float(), min_val, max_val)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()

        t_start = time.time()
        preds_valid, _, _ = predict(model, validation_data_loader, True, gpu_id)
        preds_valid_unnorm = unnormalize_labels(preds_valid, min_val, max_val)
        labels_valid_unnorm = unnormalize_labels(labels_valid, min_val, max_val)
        valid_error = mean_qerror(preds_valid_unnorm, labels_valid_unnorm)
        valid_time += (time.time() - t_start)

        logger.info("Epoch {}, train loss {}, valid loss: {}".format(epoch, loss_total / len(train_data_loader), valid_error))

    train_time = time.time() - t
    train_time = train_time - valid_time
    logger.info("Training time: {}".format(train_time))
    logger.info("Vaidation time: {}".format(valid_time))

    num_params = num_model_parameters(model)
    logger.info(f'# model parameters = {num_params}')
    logger.info(f'Size in MB = {num_params*4/1024/1024}')
    
    # Get final training and validation set predictions
    preds_train, t_total, times = predict(model, train_data_loader, gpu_id)
    logger.info("Prediction time per training sample: {}".format(t_total / len(labels_train) * 1000))

    # Unnormalize
    preds_train_unnorm = unnormalize_labels(preds_train, min_val, max_val)
    labels_train_unnorm = unnormalize_labels(labels_train, min_val, max_val)

    # Print metrics
    logger.info("\nQ-Error training set:")
    log_qerror(preds_train_unnorm, labels_train_unnorm, logger)

    return model


def test(model, test_file_path, md_dict, batch_size, output_path, logger, dbname, gpu_id, use_string, no_alias):
    dicts = md_dict["dicts"]
    column_min_max_vals = md_dict["column_min_max_vals"]
    min_val = md_dict["min_val"]
    max_val = md_dict["max_val"]

    sep, string_columns, minmax_file_path, sample_file_path, word_vectors_path, alias_dict = prepare_loading(test_file_path, dbname)
    # Load test data
    joins, predicates, tables, samples, label = load_data_from_path(test_file_path, sample_file_path, sep, no_alias, alias_dict)
    num_total_queries = len(label)

    table2vec, column2vec, op2vec, join2vec = dicts

    schema_joins = [k for k,v in join2vec.items()]
    joins, added_tables, invalid = transfer_to_canonical_form(schema_joins, joins)

    label_total = label[:]
    for index in sorted(invalid, reverse=True):
        del joins[index]
        del predicates[index]
        del tables[index]
        del samples[index]
        del label[index]
        del added_tables[index]
    print(f'number of invalid queries = {len(invalid)}')
    added_tables = convert_tables_reverse(added_tables, alias_dict)

    # Get feature encoding and proper normalization
    tables_test, samples_test = encode_samples(tables, samples, table2vec, added_tables)
    if use_string:
        is_imdb = ("job" in dbname) or ("imdb" in dbname)
        predicates_test, joins_test = encode_data_with_string(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec, string_columns, word_vectors_path, is_imdb)
    else:    
        predicates_test, joins_test = encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec)
    labels_test, _, _ = normalize_labels(label, min_val, max_val)

    logger.info("Number of test samples: {}".format(len(labels_test)))

    # Get test set predictions
    test_data = make_dataset(tables_test, samples_test, predicates_test, joins_test, labels_test)
    test_data_loader = DataLoader(test_data, batch_size=1)

    preds_test, t_total, times = predict(model, test_data_loader, gpu_id)
    logger.info("Prediction time per test sample: {}".format(t_total / len(labels_test) * 1000))

    # Unnormalize
    preds_test_unnorm = unnormalize_labels(preds_test, min_val, max_val)

    # Print metrics
    logger.info("\nQ-Error :")
    qerror = log_qerror(preds_test_unnorm, label, logger)

    # Write predictions
    with open(output_path, "w") as f:
        f.write("errs,est_cards,true_cards,query_dur_ms\n")
        j=0
        for i in range(0,num_total_queries):
            if i in invalid:
                f.write(f"-1,-1,{label_total[i]},-1\n") 
                continue
            f.write("{},{},{},{}\n".format(qerror[j], preds_test_unnorm[j], label[j], times[j] * 1000))
            j += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="path to traiing data file", type = str, default ="test-only")
    parser.add_argument("--test", help="path to test data file", type = str, default ="train-only")
    parser.add_argument("--output", help="path to output result file", type = str, default =None)
    parser.add_argument("--dbname", help="database name (imdb-small, imdb-medium, ..., default: imdb-small)", type = str, default ="imdb-small")
    parser.add_argument("--string", help="use string embeddings", action="store_true")

    parser.add_argument("--load", help="path to model file ", type = str, default ="")

    parser.add_argument("--epochs", help="number of epochs (default: 100)", type=int, default=100)
    parser.add_argument("--batch", help="batch size (default: 1024)", type=int, default=1024)
    parser.add_argument("--hid", help="number of hidden units (default: 256)", type=int, default=256)
    parser.add_argument("--lr", help="learning rate (default: 0.001)", type=float, default=0.001)
    
    parser.add_argument("--num_train", help="number of training queries (default: 100000)", type=int, default=100000)
    parser.add_argument("--num_valid", help="number of validation queries (default: 10000)", type=int, default=10000)
    parser.add_argument("--gpu", help="gpu id to use (default: 0)", type=int, default=0)

    parser.add_argument("--no_alias", help="do not use table alias in query", action="store_true")
    
    args = parser.parse_args()

    test_file_basename = os.path.basename(args.test)
    test_name = os.path.splitext(test_file_basename)[0]

    # exp_name = "{}_{}_{}".format(train_name, test_name, time.strftime("%Y%m%d-%H%M%S"))
    exp_name = "{}_{}".format(args.dbname, test_name)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    log_path = "/dev/null"
    # log_path = "logs/fcn_" + exp_name + "_" + timestamp + ".log"
    # os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ])
    logger = logging.getLogger(__name__)
    logger.info("Input args: %r", args)

    if len(args.load) > 0:
        model_path = args.load
        logger.info("load model from {}".format(model_path))
        model_dir = os.path.dirname(model_path)
        md_dict_path = model_dir + "/metadata_dict.pkl"
        md_dict_file = open(md_dict_path, "rb")
        md_dict = pickle.load(md_dict_file)
        model = torch.load(args.load, map_location=torch.device('cpu'))
    else:
        assert(args.train != "test-only")
        dicts, column_min_max_vals, min_val, max_val, labels_train, labels_valid, train_data, validation_data = get_train_dataset_from_path(
        args.train, args.dbname, args.num_train, args.num_valid, args.string)

        md_dict = dict()
        md_dict["dicts"] = dicts
        md_dict["column_min_max_vals"] = column_min_max_vals
        md_dict["min_val"] = min_val
        md_dict["max_val"] = max_val

        model = train(train_data, validation_data, md_dict, labels_train, labels_valid, args.epochs, args.batch, args.hid, args.lr, logger, args.gpu, args.string)

    if args.test != "train-only":
        if args.output == None:
            test_output_path = "results/fcn_" + exp_name + "_" + timestamp + ".csv"
        else:
            test_output_path = args.output
        test(model, args.test, md_dict, args.batch, test_output_path, logger, args.dbname, args.gpu, args.string, args.no_alias)
        logger.info(f'results file: {test_output_path}')

if __name__ == "__main__":
    main()
