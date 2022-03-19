from math import ceil
from src.feature_extraction.database_loader_new import *
from src.training.train_and_test import *
from src.internal_parameters import *
import argparse
import os
import logging
import pickle
import time

def unnormalize_list(vecs, mini, maxi):
    return [np.exp(v * (maxi - mini) + mini) for v in vecs]

def get_qerror(preds, targets):
    qerror = []
    for i in range(len(targets)):
        if (preds[i] < 1): preds[i] = 1
        if (preds[i] > targets[i]):
            qerror.append(preds[i]/targets[i])
        else:
            qerror.append(targets[i]/preds[i])
    return qerror

def log_qerror(preds, targets, logger):
    qerror = []
    for i in range(len(targets)):
        if (preds[i] < 1): preds[i] = 1
        if (preds[i] > targets[i]):
            qerror.append(preds[i]/targets[i])
        else:
            qerror.append(targets[i]/preds[i])
    logger.info("Mean: {}".format(np.mean(qerror)))
    logger.info("Min: {}".format(np.min(qerror)))
    logger.info("Median: {}".format(np.median(qerror)))
    logger.info("90th percentile: {}".format(np.percentile(qerror, 90)))
    logger.info("95th percentile: {}".format(np.percentile(qerror, 95)))
    logger.info("99th percentile: {}".format(np.percentile(qerror, 99)))
    logger.info("Max: {}".format(np.max(qerror)))

def write_results(output_path,qerror,est_card, true_card, inference_time):
    with open(output_path, "w") as f:
        f.write("errs,est_cards,true_cards,query_dur_ms\n")
        for i in range(len(est_card)):
            f.write("{},{},{},{}\n".format(qerror[i], est_card[i], true_card[i], inference_time[i] * 1000))

def test(model, test_start, test_end,plan_batches, gpu_id):
    model.cuda(gpu_id)
    model.eval()
    # cost_loss_total = 0.
    card_loss_total = 0.
    estimate_cardiality_total = []
    target_cardiality_total = []
    inference_time_total = []
    
    for batch_idx in range(test_start, test_end):
        target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = plan_batches[batch_idx]
        target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality),torch.FloatTensor(operatorss),torch.FloatTensor(extra_infoss),torch.FloatTensor(condition1ss),torch.FloatTensor(condition2ss), torch.FloatTensor(sampless), torch.FloatTensor(condition_maskss), torch.FloatTensor(mapping)
        
        operatorss, extra_infoss, condition1ss, condition2ss, condition_maskss = operatorss.squeeze(0), extra_infoss.squeeze(0), condition1ss.squeeze(0), condition2ss.squeeze(0), condition_maskss.squeeze(0).unsqueeze(2)
        sampless = sampless.squeeze(0)
        mapping = mapping.squeeze(0)

        target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = target_cost.cuda(gpu_id), target_cardinality.cuda(gpu_id), operatorss.cuda(gpu_id), extra_infoss.cuda(gpu_id), condition1ss.cuda(gpu_id), condition2ss.cuda(gpu_id), sampless.cuda(gpu_id), condition_maskss.cuda(gpu_id), mapping.cuda(gpu_id)
        target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = Variable(target_cost), Variable(target_cardinality), Variable(operatorss), Variable(extra_infoss), Variable(condition1ss), Variable(condition2ss)
        sampless = Variable(sampless)
        t_start = time.time()
        estimate_cardinality = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping)
        inference_time = time.time() - t_start
        estimate_cardinality = [float(v) for v in estimate_cardinality ]
        estimate_cardiality_total += estimate_cardinality
        inference_time = inference_time / len(target_cardinality)
        inference_time_total.append(inference_time)
        inference_time *= len(target_cardinality)
        
    return estimate_cardiality_total, inference_time_total

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input query json file", type = str)
    parser.add_argument("--output", help="path to output result file", type = str, default =None)
    parser.add_argument("--model", help="path to model file", type = str)
    parser.add_argument("--batch", help="inference batch size", type = int, default=1)
    parser.add_argument("--gpu", help="id of gpu to use", type = int, default=0)
    # parser.add_argument("--dbname", help="database name (imdb-small, imdb-medium, ..., default: imdb-small)", type = str, default ="imdb-small")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    model_path = args.model
    # dbname = args.dbname

    batch = args.batch

    # test_file_basename = os.path.basename(input_path)
    # test_name = os.path.splitext(test_file_basename)[0]
    # timestamp = time.strftime("%Y%m%d-%H%M%S")

    log_path = "/dev/null"
    # log_path = "logs/e2e_test_" + test_name + "_" + dbname + "_" + timestamp + ".log"
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

    #load meta data and model
    logger.info("load model from {}".format(model_path))
    
    model_dir = os.path.dirname(model_path)
    model = torch.load(model_path, map_location=torch.device('cpu'))
    parameters_file_path = f'{model_dir}/parameters.pkl'
    parameters_file = open(parameters_file_path, "rb")
    parameters = pickle.load(parameters_file)

    plan_batches = encode_plan_seq(input_path, parameters, batch)
    target_cards = get_target_cards(input_path)
    num_queries = len(target_cards)
    num_test_batch = ceil(num_queries / batch)
    
    est_cards, inference_times = test(model, 0, num_test_batch,plan_batches,args.gpu)

    card_label_min, card_label_max = parameters.card_label_min, parameters.card_label_max
    print(card_label_min)
    print(card_label_max)

    est_cards_unnorm = unnormalize_list(est_cards, card_label_min, card_label_max)
    log_qerror(est_cards_unnorm, target_cards, logger)
    avg_pred_time = sum(inference_times) / len(inference_times)
    logger.info(f'avg. prediction time = {avg_pred_time * 1000}ms')
    qerror = get_qerror(est_cards_unnorm, target_cards)

    results_file_path = output_path
    write_results(results_file_path, qerror, est_cards_unnorm, target_cards, inference_times)
    
if __name__ == "__main__":
    main()
