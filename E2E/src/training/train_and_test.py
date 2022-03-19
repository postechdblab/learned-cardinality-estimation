from src.plan_encoding.spilling2disk import *
from src.training.representation_model import *
from src.training.vector_loader import *

import json
import torch
import time
from torch.autograd import Variable

def unnormalize(vecs, mini, maxi):
    return torch.exp(vecs * (maxi - mini) + mini)

def get_target_cards(path):
    target_preds = []
    with open(path, 'r') as f:
        for idx, seq in enumerate(f.readlines()):
            plan = json.loads(seq)
            target_preds.append(float(plan['cardinality']))
    return target_preds

def encode_plan_seq(path, parameters, batch_size=64, max_queries = -1):
    test_plans = []
    with open(path, 'r') as f:
        for idx, seq in enumerate(f.readlines()):
            plan = json.loads(seq)
            test_plans.append(plan)
            #     shuffle(test_plans)
    return get_data_job(plans=test_plans, parameters = parameters, batch_size=batch_size, max_queries=max_queries)


def encode_train_plan_seq_save(path, parameters, batch_size, directory,  max_queries = -1):
    test_plans = []
    with open(path, 'r') as f:
        for idx, seq in enumerate(f.readlines()):
            plan = json.loads(seq)
            test_plans.append(plan)
            #     shuffle(test_plans)
    save_data_job(plans=test_plans, parameters = parameters, batch_size=batch_size, directory=directory, max_queries = max_queries)


def encode_test_plan_seq_save(path, parameters, batch_size=64, directory='/home/sunji/learnedcardinality/job'):
    test_plans = []
    with open(path, 'r') as f:
        for idx, seq in enumerate(f.readlines()):
            plan = json.loads(seq)
            test_plans.append(plan)
            #     shuffle(test_plans)
    save_data_job(plans=test_plans, parameters = parameters, istest=True, batch_size=batch_size, directory=directory)


def qerror_loss(preds, targets, mini, maxi):
    qerror = []
    preds = unnormalize(preds, mini, maxi)
    targets = unnormalize(targets, mini, maxi)
    for i in range(len(targets)):
        if preds[i] < 1: preds[i] = 1
        if (preds[i] > targets[i]).cpu().data.numpy()[0]:
            qerror.append(preds[i] / targets[i])
        else:
            qerror.append(targets[i] / preds[i])
    return torch.mean(torch.cat(qerror)), torch.median(torch.cat(qerror)), torch.max(torch.cat(qerror)), torch.argmax(
        torch.cat(qerror))

def train(train_start, train_end, validate_start, validate_end, num_epochs, parameters, plan_batches, model_dir_path, gpu_id):
    input_dim = parameters.condition_op_dim
    hidden_dim = parameters.hid1
    hid_dim = parameters.hid2
    extra_info_size = max(parameters.column_total_num, parameters.index_total_num)
    # middle_result_dim = 128
    # task_num = 2
    model = Representation(input_dim, hidden_dim, hid_dim, extra_info_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.cuda(gpu_id)
    model.train()
    start = time.time()

    validation_time = 0.0
    model_save_time = 0.0

    for epoch in range(num_epochs):
        # cost_loss_total = 0.
        card_loss_total = 0.
        model.train()
        for batch_idx in range(train_start, train_end):
            # print('batch_idx: ', batch_idx)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = plan_batches[batch_idx]
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = torch.FloatTensor(
                target_cost), torch.FloatTensor(target_cardinality), torch.FloatTensor(operatorss), torch.FloatTensor(
                extra_infoss), torch.FloatTensor(condition1ss), torch.FloatTensor(condition2ss), torch.FloatTensor(
                sampless), torch.FloatTensor(condition_maskss), torch.FloatTensor(mapping)
            operatorss, extra_infoss, condition1ss, condition2ss, condition_maskss = operatorss.squeeze(
                0), extra_infoss.squeeze(0), condition1ss.squeeze(0), condition2ss.squeeze(0), condition_maskss.squeeze(
                0).unsqueeze(2)
            sampless = sampless.squeeze(0)
            mapping = mapping.squeeze(0)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = target_cost.cuda(gpu_id), target_cardinality.cuda(gpu_id), operatorss.cuda(gpu_id), extra_infoss.cuda(gpu_id), condition1ss.cuda(gpu_id), condition2ss.cuda(gpu_id)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = Variable( target_cost), Variable(target_cardinality), Variable(operatorss), Variable(extra_infoss), Variable( condition1ss), Variable(condition2ss)
            sampless = sampless.cuda(gpu_id)
            sampless = Variable(sampless)
            optimizer.zero_grad()
            condition_maskss, mapping = condition_maskss.cuda(gpu_id), mapping.cuda(gpu_id)
            estimate_cardinality = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless,
                                                        condition_maskss, mapping)
            # target_cost = target_cost
            target_cardinality = target_cardinality
            # cost_loss, cost_loss_median, cost_loss_max, cost_max_idx = qerror_loss(estimate_cost, target_cost, parameters.cost_label_min, parameters.cost_label_max)
            card_loss, card_loss_median, card_loss_max, card_max_idx = qerror_loss(estimate_cardinality, target_cardinality, parameters.card_label_min,
                                                                                   parameters.card_label_max)
            loss =  card_loss
            # cost_loss_total += cost_loss.item()
            card_loss_total += card_loss.item()
            loss.backward()
            optimizer.step()
        batch_num = train_end - train_start
        train_loss = card_loss_total / batch_num

        card_loss_total = 0.
        t_start = time.time()
        for batch_idx in range(validate_start, validate_end):
            # target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = get_batch_job( batch_idx, directory=directory)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = plan_batches[batch_idx]
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = torch.FloatTensor(
                target_cost), torch.FloatTensor(target_cardinality), torch.FloatTensor(operatorss), torch.FloatTensor(
                extra_infoss), torch.FloatTensor(condition1ss), torch.FloatTensor(condition2ss), torch.FloatTensor(
                sampless), torch.FloatTensor(condition_maskss), torch.FloatTensor(mapping)
            operatorss, extra_infoss, condition1ss, condition2ss, condition_maskss = operatorss.squeeze(
                0), extra_infoss.squeeze(0), condition1ss.squeeze(0), condition2ss.squeeze(0), condition_maskss.squeeze(
                0).unsqueeze(2)
            sampless = sampless.squeeze(0)
            mapping = mapping.squeeze(0)
            
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = target_cost.cuda(gpu_id), target_cardinality.cuda(gpu_id), operatorss.cuda(gpu_id), extra_infoss.cuda(gpu_id), condition1ss.cuda(gpu_id), condition2ss.cuda(gpu_id)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = Variable(
                target_cost), Variable(target_cardinality), Variable(operatorss), Variable(extra_infoss), Variable(
                condition1ss), Variable(condition2ss)
            
            sampless = sampless.cuda(gpu_id)
            sampless = Variable(sampless)
            condition_maskss, mapping = condition_maskss.cuda(gpu_id), mapping.cuda(gpu_id)

            estimate_cardinality = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless,
                                                        condition_maskss, mapping)
            target_cardinality = target_cardinality
            # cost_loss, cost_loss_median, cost_loss_max, cost_max_idx = qerror_loss(estimate_cost, target_cost, parameters.cost_label_min, parameters.cost_label_max)
            card_loss, card_loss_median, card_loss_max, card_max_idx = qerror_loss(estimate_cardinality, target_cardinality, parameters.card_label_min, parameters.card_label_max)
            loss =  card_loss
            card_loss_total += card_loss.item()
        batch_num = validate_end - validate_start
        valid_loss = card_loss_total / batch_num

        validation_time += time.time() - t_start

        parameters.logger.info("Epoch {}: training loss: {}, validation loss: {}".format(epoch, train_loss, valid_loss))
        model_path = model_dir_path + "/model_" + str(epoch+1)
        t_start = time.time()
        torch.save(model, model_path)
        model_save_time = time.time() - t_start
    total_time = time.time() - start

    parameters.logger.info("Training time = {} sec".format(total_time - validation_time - model_save_time))
    parameters.logger.info("Validation time = {} sec".format(validation_time ))
    parameters.logger.info("Model save time = {} sec".format(model_save_time))


    return model


def load_and_train(train_start, train_end, validate_start, validate_end, num_epochs, parameters, encode_dir_path, model_dir_path, gpu_id):
    input_dim = parameters.condition_op_dim
    hidden_dim = parameters.hid1
    hid_dim = parameters.hid2
    extra_info_size = max(parameters.column_total_num, parameters.index_total_num)
    # middle_result_dim = 128
    # task_num = 2
    model = Representation(input_dim, hidden_dim, hid_dim, extra_info_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.cuda(gpu_id)
    model.train()
    start = time.time()

    validation_time = 0.0
    model_save_time = 0.0
    load_time = 0.0

    for epoch in range(num_epochs):
        # cost_loss_total = 0.
        card_loss_total = 0.
        model.train()
        for batch_idx in range(train_start, train_end):
            print('batch_idx: ', batch_idx)
            
            t_start = time.time()
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = get_batch_job(batch_idx, encode_dir_path)
            load_time += time.time() - t_start
            print(f'load time = {time.time() - t_start}')
            t_start = time.time()
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = torch.FloatTensor(
                target_cost), torch.FloatTensor(target_cardinality), torch.FloatTensor(operatorss), torch.FloatTensor(
                extra_infoss), torch.FloatTensor(condition1ss), torch.FloatTensor(condition2ss), torch.FloatTensor(
                sampless), torch.FloatTensor(condition_maskss), torch.FloatTensor(mapping)
            operatorss, extra_infoss, condition1ss, condition2ss, condition_maskss = operatorss.squeeze(
                0), extra_infoss.squeeze(0), condition1ss.squeeze(0), condition2ss.squeeze(0), condition_maskss.squeeze(
                0).unsqueeze(2)
            sampless = sampless.squeeze(0)
            mapping = mapping.squeeze(0)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = target_cost.cuda(gpu_id), target_cardinality.cuda(gpu_id), operatorss.cuda(gpu_id), extra_infoss.cuda(gpu_id), condition1ss.cuda(gpu_id), condition2ss.cuda(gpu_id)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = Variable( target_cost), Variable(target_cardinality), Variable(operatorss), Variable(extra_infoss), Variable( condition1ss), Variable(condition2ss)
            sampless = sampless.cuda(gpu_id)
            sampless = Variable(sampless)
            optimizer.zero_grad()
            condition_maskss, mapping = condition_maskss.cuda(gpu_id), mapping.cuda(gpu_id)
            estimate_cardinality = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless,
                                                        condition_maskss, mapping)
            # target_cost = target_cost
            target_cardinality = target_cardinality
            # cost_loss, cost_loss_median, cost_loss_max, cost_max_idx = qerror_loss(estimate_cost, target_cost, parameters.cost_label_min, parameters.cost_label_max)
            card_loss, card_loss_median, card_loss_max, card_max_idx = qerror_loss(estimate_cardinality, target_cardinality, parameters.card_label_min,
                                                                                   parameters.card_label_max)
            loss =  card_loss
            # cost_loss_total += cost_loss.item()
            card_loss_total += card_loss.item()
            loss.backward()
            optimizer.step()
            print(f'train time = {time.time() - t_start}')
            # free_batch([target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping])
        batch_num = train_end - train_start
        train_loss = card_loss_total / batch_num

        card_loss_total = 0.
        t_start = time.time()
        for batch_idx in range(validate_start, validate_end):
            # target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = get_batch_job( batch_idx, directory=directory)
            # target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = plan_batches[batch_idx]
            t_start = time.time()
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = get_batch_job(batch_idx, encode_dir_path)
            load_time += time.time() - t_start
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = torch.FloatTensor(
                target_cost), torch.FloatTensor(target_cardinality), torch.FloatTensor(operatorss), torch.FloatTensor(
                extra_infoss), torch.FloatTensor(condition1ss), torch.FloatTensor(condition2ss), torch.FloatTensor(
                sampless), torch.FloatTensor(condition_maskss), torch.FloatTensor(mapping)
            operatorss, extra_infoss, condition1ss, condition2ss, condition_maskss = operatorss.squeeze(
                0), extra_infoss.squeeze(0), condition1ss.squeeze(0), condition2ss.squeeze(0), condition_maskss.squeeze(
                0).unsqueeze(2)
            sampless = sampless.squeeze(0)
            mapping = mapping.squeeze(0)
            
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = target_cost.cuda(gpu_id), target_cardinality.cuda(gpu_id), operatorss.cuda(gpu_id), extra_infoss.cuda(gpu_id), condition1ss.cuda(gpu_id), condition2ss.cuda(gpu_id)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = Variable(
                target_cost), Variable(target_cardinality), Variable(operatorss), Variable(extra_infoss), Variable(
                condition1ss), Variable(condition2ss)
            
            sampless = sampless.cuda(gpu_id)
            sampless = Variable(sampless)
            condition_maskss, mapping = condition_maskss.cuda(gpu_id), mapping.cuda(gpu_id)

            estimate_cardinality = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless,
                                                        condition_maskss, mapping)
            target_cardinality = target_cardinality
            # cost_loss, cost_loss_median, cost_loss_max, cost_max_idx = qerror_loss(estimate_cost, target_cost, parameters.cost_label_min, parameters.cost_label_max)
            card_loss, card_loss_median, card_loss_max, card_max_idx = qerror_loss(estimate_cardinality, target_cardinality, parameters.card_label_min, parameters.card_label_max)
            loss =  card_loss
            card_loss_total += card_loss.item()
        batch_num = validate_end - validate_start
        valid_loss = card_loss_total / batch_num

        validation_time += time.time() - t_start

        parameters.logger.info("Epoch {}: training loss: {}, validation loss: {}".format(epoch, train_loss, valid_loss))
        model_path = model_dir_path + "/model_" + str(epoch+1)
        t_start = time.time()
        torch.save(model, model_path)
        model_save_time = time.time() - t_start
    total_time = time.time() - start

    parameters.logger.info("Training time = {} sec".format(total_time - validation_time - model_save_time - load_time))
    parameters.logger.info("Validation time = {} sec".format(validation_time ))
    parameters.logger.info("Model save time = {} sec".format(model_save_time))
    parameters.logger.info("Batch load time = {} sec".format(load_time))


    return model
