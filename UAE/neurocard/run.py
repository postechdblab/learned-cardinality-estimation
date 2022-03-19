"""Tune-integrated training script for parallel experiments."""
import requests
import argparse
import collections
import gc
import glob
import os
import pickle
import pprint
import time
import random
import shutil
import sys
import math

import numpy as np
import pandas as pd
import ray
from ray import tune
import psutil
from ray.tune import logger as tune_logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.tune.schedulers import ASHAScheduler
from torch.utils import data
import wandb

import common
import datasets
import estimators as estimators_lib
import experiments
import factorized_sampler
import fair_sampler
import join_utils
import made
import train_utils
import transformer
import utils

# +@ import datetime
import datetime
import traceback
from train_utils import print_gpu_info, print_tensor_mem


torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()

parser.add_argument('--run',
                    nargs='+',
                    default=experiments.TEST_CONFIGS.keys(),
                    type=str,
                    required=False,
                    help='List of experiments to run.')
# Resources per trial.
parser.add_argument('--cpus',
                    default=1,
                    type=int,
                    required=False,
                    help='Number of CPU cores per trial.')
parser.add_argument(
    '--gpus',
    default=1,
    type=int,
    required=False,
    help='Number of GPUs per trial. No effect if no GPUs are available.')


# +@ add arguments
parser.add_argument(
    '--workload',
    default='',
    type=str,
    required=False,
    help='specify workload name')

parser.add_argument(
    '--log_mode',
    default=False,
    type=bool,
    required=False,
    help='save training tuple mode')
parser.add_argument(
    '--tuning',
    default=False,
    type=bool,
    required=False,
    help='tuning')

parser.add_argument(
    '--loss_file',
    default='loss_result.csv',
    type=str,
    help='loss file path')


args = parser.parse_args()

cur_workload = args.workload
MODEL_PATH = '/mnt/disk2/models'
# +@ load argument
log_mode = args.log_mode
available_dataset = datasets.dataset_list

def auto_garbage_collect(pct=80.0):
    """
    auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                              This is called to deal with an issue in Ray not freeing up used memory.

        pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
    """
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
    return

class DataParallelPassthrough(torch.nn.DataParallel):
    """Wraps a model with nn.DataParallel and provides attribute accesses."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def TotalGradNorm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item()**norm_type
    total_norm = total_norm**(1. / norm_type)
    return total_norm

def get_qerror_torch(est_card, card):
    est_card[est_card<1] = 1.0
    card[card==0] = 1.0
    # if torch.isnan(est_card / card).any() or torch.isnan(card / est_card).any():
    #     print(torch.isnan(card).any())
    #     print(torch.isnan(est_card).any())
    #     print(torch.isnan(est_card / card).any())
    #     print(torch.isnan(card / est_card).any())
    qerror_batch = torch.max(est_card / card, card / est_card)

    qerror_batch[card <= 0] = est_card[card <= 0]
    qerror_batch[est_card <= 0] = card[est_card <= 0]
    if torch.isnan(qerror_batch).any():
        return torch.tensor(np.nanmean(qerror_batch.cpu()),device=qerror_batch.device)
    return torch.mean(qerror_batch)


def run_epoch(split,
              model,
              opt,
              train_data,
              val_data=None,
              batch_size=100,
              upto=None,
              epoch_num=None,
              epochs=1,
              verbose=False,
              log_every=10,
              return_losses=False,
              table_bits=None,
              warmups=1000,
              loader=None,
              query_driven=False,
              query_only=False,
              query_driven_estimator=None,
              #query_driven_dataset=None,
              query_list=None,
              card_list=None,
              q_bs=None,
              q_weight=0,
              constant_lr=None,
              use_meters=True,
              summary_writer=None,
              lr_scheduler=None,
              custom_lr_lambda=None,
              label_smoothing=0.0,
              neurocard_instance=None,
              sep_backward = False): # +@ for save last loss
    if neurocard_instance is not None :
        accum_iter = neurocard_instance.accum_iter
        max_step = neurocard_instance.max_steps
    else:
        accum_iter = 1
    torch.set_grad_enabled(split == 'train')
    model.train() if split == 'train' else model.eval()
    dataset = train_data if split == 'train' else val_data
    losses = []
    d_losses = []
    q_losses = []

    # +@ for save training tuple, change join iter datasets' attribute
    if isinstance(dataset,factorized_sampler.FactorizedSamplerIterDataset) and (split == 'train') and log_mode :
        print('--- Logging Train Tuple ')
        dataset.join_iter_dataset.SetLogTrain(True)


    if loader is None:
        loader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=(split == 'train'))

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, 'orderings'):
        nsamples = len(model.orderings)
        if verbose:
            print('setting nsamples to', nsamples)

    dur_meter = train_utils.AverageMeter(
        'dur', lambda v: '{:.0f}s'.format(v), display_average=False)
    lr_meter = train_utils.AverageMeter('lr', ':.5f', display_average=False)
    tups_meter = train_utils.AverageMeter('tups',
                                          utils.HumanFormat,
                                          display_average=False)
    loss_meter = train_utils.AverageMeter('loss (bits/tup)', ':.2f')
    #query_loss_meter = train_utils.AverageMeter('query loss (bits)', ':.2f')
    train_throughput = train_utils.AverageMeter('tups/s',
                                                utils.HumanFormat,
                                                display_average=False)
    batch_time = train_utils.AverageMeter('sgd_ms', ':3.1f')
    data_time = train_utils.AverageMeter('data_ms', ':3.1f')
    progress = train_utils.ProgressMeter(upto, [
        batch_time,
        data_time,
        dur_meter,
        lr_meter,
        tups_meter,
        train_throughput,
        loss_meter,
        #query_loss_meter,
    ])

    begin_time = t1 = time.time()
    print(f"Accum step {accum_iter}")

    #if query_driven: 
    #    query_driven_dataset.Start()

    if query_driven and split == 'train':
        if q_bs is None:
            q_bs = math.ceil(len(query_list) / upto)
            q_bs = int(q_bs)
            query_iter = 0
            query_run_size = None
        else:
            query_run_size = query_driven_estimator.batch_size
            query_iter = (q_bs + query_run_size -1)//query_run_size
            
            
        print(f'# queries = {len(query_list)}, upto = {upto}, query batch size = {q_bs}, query run size = {query_iter}')
        
        query_card_list = list(zip(query_list, card_list))
        
        
        
        
        np.random.shuffle(query_card_list)
    # print_gpu_info('before step')
    for step, xb in enumerate(loader):
        # print_gpu_info('start step',step=step)
        torch.cuda.empty_cache()
        data_time.update((time.time() - t1) * 1e3)
        if split == 'train':
            if isinstance(dataset, data.IterableDataset):
                # Can't call len(loader).
                global_steps = upto * epoch_num + step + 1
            else:
                global_steps = len(loader) * epoch_num + step + 1

            if constant_lr:
                lr = constant_lr
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
            elif custom_lr_lambda:
                lr_scheduler = None
                lr = custom_lr_lambda(global_steps)
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
            elif lr_scheduler is None:
                t = warmups
                if warmups < 1:  # A ratio.
                    t = int(warmups * upto * epochs)

                d_model = model.embed_size
                lr = (d_model**-0.5) * min(
                    (global_steps**-.5), global_steps * (t**-1.5))
                for param_group in opt.param_groups:
                    param_group['lr'] = lr
            else:
                # We'll call lr_scheduler.step() below.
                lr = opt.param_groups[0]['lr']

        if upto and step >= upto:
            break

        if not query_only:
            if isinstance(xb, list):
                # This happens if using data.TensorDataset.
                assert len(xb) == 1, xb
                xb = xb[0]

            xb = xb.float().to(train_utils.get_device(), non_blocking=True)
#             print_gpu_info('DD xb',step=step)
            # Forward pass, potentially through several orderings.
            xbhat = None
            model_logits = []
            num_orders_to_forward = 1
            if split == 'test' and nsamples > 1:
                # At test, we want to test the 'true' nll under all orderings.
                num_orders_to_forward = nsamples

            for i in range(num_orders_to_forward):
                if hasattr(model, 'update_masks'):
                    # We want to update_masks even for first ever batch.
                    model.update_masks()

                model_out = model(xb)
#                 print_gpu_info('DD model out',step=step)
                model_logits.append(model_out)
                if xbhat is None:
                    xbhat = torch.zeros_like(model_out)
                xbhat += model_out

            if num_orders_to_forward == 1:
                loss = model.nll(xbhat, xb, label_smoothing=label_smoothing).mean()
#                 print_gpu_info('DD model nll',step=step)
#                 print_tensor_mem(loss,step=step,tag='DD loss tensor')
            else:
                assert False
                # Average across orderings & then across minibatch.
                #
                #   p(x) = 1/N sum_i p_i(x)
                #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                #             = log(1/N) + logsumexp ( log p_i(x) )
                #             = log(1/N) + logsumexp ( - nll_i (x) )
                #
                # Used only at test time.
                logps = []  # [batch size, num orders]
                assert len(model_logits) == num_orders_to_forward, len(model_logits)
                for logits in model_logits:
                    # Note the minus.
                    logps.append(
                        -model.nll(logits, xb, label_smoothing=label_smoothing))
                logps = torch.stack(logps, dim=1)
                logps = logps.logsumexp(dim=1) + torch.log(
                    torch.tensor(1.0 / nsamples, device=logps.device))
                loss = (-logps).mean()
        else:
            loss = torch.tensor(0., device=train_utils.get_device())

        if split == 'train' and query_driven:
            if sep_backward:
                if step%200 ==0:
                    print("[sep_backward] - dd backward")
                loss.backward()
                d_losses.append(loss.detach().cpu().item())



            torch.cuda.empty_cache()
            #queries, cards = query_driven_dataset.GetBatch()
            if step * q_bs >= len(query_list):
                q_c_tmp = query_card_list[step * q_bs - len(query_list): (step + 1) * q_bs - len(query_list)]
            elif (step + 1) * q_bs > len(query_list):
                q_c_tmp = query_card_list[step * q_bs:] + query_card_list[0: (step + 1) * q_bs - len(query_list)]
            else:
                q_c_tmp = query_card_list[step * q_bs: (step + 1) * q_bs]

            train_queries = [q for q, c in q_c_tmp]
            train_cards = [c for q, c in q_c_tmp]
            train_cards = np.array(train_cards)
            train_cards = torch.as_tensor(train_cards, dtype=torch.float32)
            train_cards = train_cards.to(train_utils.get_device())
            cols_list = [query[0] for i, query in enumerate(train_queries)]
            ops_list = [query[1] for i, query in enumerate(train_queries)]
            vals_list = [query[2] for i, query in enumerate(train_queries)]

            if query_iter == 1:
                est_card_batch = query_driven_estimator.Query(cols_list, ops_list, vals_list).detach().cpu()
                if torch.isnan(est_card_batch).any():
                    continue
                q_loss = get_qerror_torch(est_card_batch.to(train_utils.get_device()), train_cards)
            elif query_iter > 0:
                est_card_batch_list = list()
                for i in range(query_iter):
                    if i == (query_iter-1): #last step
                        est_card_batch = query_driven_estimator.Query(cols_list[i*query_run_size:], ops_list[i*query_run_size:], vals_list[i*query_run_size:]).detach().cpu()
                        est_card_batch_list.append(est_card_batch)
                        break
                    est_card_batch = query_driven_estimator.Query(cols_list[i*query_run_size:(i+1)*query_run_size], ops_list[i*query_run_size:(i+1)*query_run_size], vals_list[i*query_run_size:(i+1)*query_run_size]).detach().cpu()
                    est_card_batch_list.append(est_card_batch)

                est_card_batch = torch.cat(est_card_batch_list)
                if torch.isnan(est_card_batch).any():
                    continue
                q_loss = get_qerror_torch(est_card_batch.to(train_utils.get_device()), train_cards)

            else:
                assert False
            if sep_backward:
                # loss = torch.tensor(0., device=train_utils.get_device())
                q_losses.append(q_loss.detach().item())
                loss = q_weight * q_loss
            else :
                d_losses.append(loss.detach().item())
                q_losses.append(q_loss.detach().item())
                loss = q_weight * q_loss + loss

        if sep_backward:
            losses.append(loss.detach().item() + d_losses[-1])
        else:
            losses.append(loss.detach().item())

        if split == 'train':
            if step == 0:
                opt.zero_grad()
            if query_only or sep_backward:
                loss.requires_grad = True
                loss.backward()
            elif query_driven:
                loss.backward()
            else:
                loss.backward()
            l2_grad_norm = TotalGradNorm(model.parameters())
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1, norm_type=2)
            # opt.step()
            if ((step + 1) % accum_iter == 0) or (step + 1 == max_step):
                opt.step()
                opt.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step()
            loss_bits = loss.item() / np.log(2)
            # Number of tuples processed in this epoch so far.
            ntuples = (step + 1) * batch_size
            if use_meters:
                dur = time.time() - begin_time
                lr_meter.update(lr)
                tups_meter.update(ntuples)
                loss_meter.update(loss_bits)
                dur_meter.update(dur)
                train_throughput.update(ntuples / dur)

            if summary_writer is not None:
                summary_writer.add_scalar('train/lr',
                                          lr,
                                          global_step=global_steps)
                summary_writer.add_scalar('train/tups',
                                          ntuples,
                                          global_step=global_steps)
                summary_writer.add_scalar('train/tups_per_sec',
                                          ntuples / dur,
                                          global_step=global_steps)
                summary_writer.add_scalar('train/nll',
                                          loss_bits,
                                          global_step=global_steps)

            if step % log_every == 0:
                if table_bits:
                    print(
                            'Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f} {:.5f} lr, {} tuples seen ({} tup/s)'
                        .format(
                            epoch_num, step, split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2), table_bits, lr,
                            utils.HumanFormat(ntuples),
                            utils.HumanFormat(ntuples /
                                              (time.time() - begin_time))))
                elif not use_meters:
                    print(
                        'Epoch {} Iter {}, {} loss {:.3f} bits/tuple, {:.5f} lr'
                        .format(epoch_num, step, split,
                                loss.item() / np.log(2), lr))

        loss = loss.detach().cpu()

        
        if verbose:
            print('%s epoch average loss: %f' % (split, np.mean(losses)))

        batch_time.update((time.time() - t1) * 1e3)
        t1 = time.time()
        if split == 'train' and step % log_every == 0 and use_meters:
            progress.display(step)

    # +@ for save model state
    if neurocard_instance is not None:
        neurocard_instance.lloss = loss
        neurocard_instance.d_loss = np.mean(d_losses)
        neurocard_instance.q_loss = np.mean(q_losses)
    if return_losses:
        return losses
    return np.mean(losses)


def MakeMade(
        table,
        scale,
        layers,
        cols_to_train,
        seed,
        factor_table=None,
        fixed_ordering=None,
        special_orders=0,
        order_content_only=True,
        order_indicators_at_front=True,
        inv_order=True,
        residual=True,
        direct_io=True,
        input_encoding='embed',
        output_encoding='embed',
        embed_size=32,
        dropout=True,
        grouped_dropout=False,
        per_row_dropout=False,
        fixed_dropout_ratio=False,
        input_no_emb_if_leq=False,
        embs_tied=True,
        resmade_drop_prob=0.,
        # Join specific:
        num_joined_tables=None,
        table_dropout=None,
        table_num_columns=None,
        table_column_types=None,
        table_indexes=None,
        table_primary_index=None,
        # DMoL
        num_dmol=0,
        scale_input=False,
        dmol_cols=[]):
    dmol_col_indexes = []
    if dmol_cols:
        for i in range(len(cols_to_train)):
            if cols_to_train[i].name in dmol_cols:
                dmol_col_indexes.append(i)
    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        layers if layers > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        num_masks=max(1, special_orders),
        natural_ordering=True,
        input_bins=[c.DistributionSize() for c in cols_to_train],
        do_direct_io_connections=direct_io,
        input_encoding=input_encoding,
        output_encoding=output_encoding,
        embed_size=embed_size,
        input_no_emb_if_leq=input_no_emb_if_leq,
        embs_tied=embs_tied,
        residual_connections=residual,
        factor_table=factor_table,
        seed=seed,
        fixed_ordering=fixed_ordering,
        resmade_drop_prob=resmade_drop_prob,

        # Wildcard skipping:
        dropout_p=dropout,
        fixed_dropout_p=fixed_dropout_ratio,
        grouped_dropout=grouped_dropout,
        learnable_unk=True,
        per_row_dropout=per_row_dropout,

        # DMoL
        num_dmol=num_dmol,
        scale_input=scale_input,
        dmol_col_indexes=dmol_col_indexes,

        # Join support.
        num_joined_tables=num_joined_tables,
        table_dropout=table_dropout,
        table_num_columns=table_num_columns,
        table_column_types=table_column_types,
        table_indexes=table_indexes,
        table_primary_index=table_primary_index,
    ).to(train_utils.get_device())

    if special_orders > 0:
        orders = []

        if order_content_only:
            print('Leaving out virtual columns from orderings')
            cols = [c for c in cols_to_train if not c.name.startswith('__')]
            inds_cols = [c for c in cols_to_train if c.name.startswith('__in_')]
            num_indicators = len(inds_cols)
            num_content, num_virtual = len(cols), len(cols_to_train) - len(cols)

            # Data: { content }, { indicators }, { fanouts }.
            for i in range(special_orders):
                rng = np.random.RandomState(i + 1)
                content = rng.permutation(np.arange(num_content))
                inds = rng.permutation(
                    np.arange(num_content, num_content + num_indicators))
                fanouts = rng.permutation(
                    np.arange(num_content + num_indicators, len(cols_to_train)))

                if order_indicators_at_front:
                    # Model: { indicators }, { content }, { fanouts },
                    # permute each bracket independently.
                    order = np.concatenate(
                        (inds, content, fanouts)).reshape(-1,)
                else:
                    # Model: { content }, { indicators }, { fanouts }.
                    # permute each bracket independently.
                    order = np.concatenate(
                        (content, inds, fanouts)).reshape(-1,)
                assert len(np.unique(order)) == len(cols_to_train), order
                orders.append(order)
        else:
            # Permute content & virtual columns together.
            for i in range(special_orders):
                orders.append(
                    np.random.RandomState(i + 1).permutation(
                        np.arange(len(cols_to_train))))

        if factor_table:
            # Correct for subvar ordering.
            for i in range(special_orders):
                # This could have [..., 6, ..., 4, ..., 5, ...].
                # So we map them back into:
                # This could have [..., 4, 5, 6, ...].
                # Subvars have to be in order and also consecutive
                order = orders[i]
                for orig_col, sub_cols in factor_table.fact_col_mapping.items():
                    first_subvar_index = cols_to_train.index(sub_cols[0])
                    print('Before', order)
                    for j in range(1, len(sub_cols)):
                        subvar_index = cols_to_train.index(sub_cols[j])
                        order = np.delete(order,
                                          np.argwhere(order == subvar_index))
                        order = np.insert(
                            order,
                            np.argwhere(order == first_subvar_index)[0][0] + j,
                            subvar_index)
                    orders[i] = order
                    print('After', order)

        print('Special orders', np.array(orders))

        if inv_order:
            for i, order in enumerate(orders):
                orders[i] = np.asarray(utils.InvertOrder(order))
            print('Inverted special orders:', orders)

        model.orderings = orders
    return model


class NeuroCard(tune.Trainable):

    def _setup(self, config):
        self.config = config
        torch.cuda.empty_cache()
        print('NeuroCard config:')
        pprint.pprint(config)



        os.chdir(config['cwd'])
        for k, v in config.items():
            setattr(self, k, v)
        if hasattr(self,'join_pred'):
            os.environ['join_pred'] = self.join_pred
        if not hasattr(self,'gumbel_tmp'):
            self.gumbel_tmp = 1.0
        if not hasattr(self,'use_query_validation'):
            self.use_query_validation = False
        if not hasattr(self,'sep_backward'):
            self.sep_backward = False
        if config['__gpu'] == 0:
            torch.set_num_threads(config['__cpu'])

        # +@ training time recoding
        self.total_train_time = 0
        self.mininum_loss = -1
        # W&B.
        # Do wandb.init() after the os.chdir() above makes sure that the Git
        # diff file (diff.patch) is w.r.t. the directory where this file is in,
        # rather than w.r.t. Ray's package dir.
        # wandb_project = config['__run']
        # now = datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
        # wandb_name = f"trial_{workload_name}_{self.fc_hiddens}_{self.embed_size}_{self.word_size_bits}_{self.layers}.tar'"
        # wandb.init(name=wandb_name,
        #            sync_tensorboard=True,
        #            config=config,
        #            project=wandb_project)

        # +@ for retrain model, load epoch number
        if ('epoch' not in dir(self)) or (self.epoch is None) :
            self.epoch = 0
        if 'random_seed' not in dir(self) :
            self.random_seed = None
        if self.random_seed is not None :
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            self.rng = np.random.RandomState(self.random_seed)
        else : self.rng = None

        # model_path_list = glob.glob(f'/mnt/disk2/models/{self.workload_name}_*_{self.epochs}_{self.fc_hiddens}_{self.embed_size}_{self.word_size_bits}_*.tar')
        # if len(model_path_list)>0:
        #     last_epoch = 0
        #     for path in model_path_list:
        #         model_epoch = int(path.split('_')[1])
        #         if last_epoch < model_epoch:
        #             last_epoch = model_epoch
        #             self.checkpoint_to_load = path

        if isinstance(self.join_tables, int):
            # Hack to support training single-model tables.
            sorted_table_names = sorted(
                list(datasets.JoinOrderBenchmark.GetJobLightJoinKeys().keys()))
            self.join_tables = [sorted_table_names[self.join_tables]]

        # Try to make all the runs the same, except for input orderings.


        # Common attributes.
        self.loader = None
        self.join_spec = None
        join_iter_dataset = None
        table_primary_index = None



        # New datasets should be loaded here.
        assert self.dataset in available_dataset # +@ assert available dataset, not only imdb

        print('Training on Join({})'.format(self.join_tables))
        loaded_tables = []
        for t in self.join_tables:
            print('Loading', t)
            # +@ we use expandable load function
            table = datasets.LoadDataset(self.dataset,t,data_dir=self.data_dir, use_cols=self.use_cols)
            table.data.info()
            loaded_tables.append(table)
        if len(self.join_tables) > 1:
            join_spec, join_iter_dataset, loader, table = self.MakeSamplerDatasetLoader(loaded_tables)
            self.join_spec = join_spec
            self.train_data = join_iter_dataset
            self.loader = loader
            # +@ using join spec data, find root
            table_primary_index = [t.name for t in loaded_tables].index(self.join_spec.join_root)
            if hasattr(join_iter_dataset.join_iter_dataset,'sampler'):
                table.cardinality = join_iter_dataset.join_iter_dataset.sampler.join_card
                print(f"True cardinality - from jct  {table.cardinality}")
            else:
                table.cardinality = datasets.get_cardinality(self.dataset,self.join_tables)
                print(f"True cardinality - {table.cardinality}")
            # +@ use expandable get card function
            # table.cardinality = datasets.get_cardinality(self.dataset,self.join_tables)
            self.train_data.cardinality = table.cardinality

            # print('rows in full join', table.cardinality,
            # 'cols in full join', len(table.columns), 'cols:', table)
        else:
            # Train on a single table.
            table = loaded_tables[0]
            print(f"True cardinality - single table {table.cardinality}")


        # if self.dataset != 'imdb' or len(self.join_tables) == 1:
        # +@ remove unnecessary condition
        if len(self.join_tables) == 1:
            join_spec = join_utils.get_single_join_spec(self.__dict__)
            self.join_spec = join_spec
            table.data.info()
            self.train_data = self.MakeTableDataset(table)

        self.table = table
        # Provide true cardinalities in a file or implement an oracle CardEst.
        self.oracle = None
        self.table_bits = 0

        # A fixed ordering?
        self.fixed_ordering = self.MakeOrdering(table)

        model = self.MakeModel(self.table,
                               self.train_data,
                               table_primary_index=table_primary_index)

        # NOTE: ReportModel()'s returned value is the true model size in
        # megabytes containing all all *trainable* parameters.  As impl
        # convenience, the saved ckpts on disk have slightly bigger footprint
        # due to saving non-trainable constants (the masks in each layer) as
        # well.  They can be deterministically reconstructed based on RNG seeds
        # and so should not be counted as model size.
        self.mb = train_utils.ReportModel(model)
        if not isinstance(model, transformer.Transformer):
            print('applying train_utils.weight_init()')
            model.apply(train_utils.weight_init)
        self.model = model

        if self.use_data_parallel:
            self.model = DataParallelPassthrough(self.model)

        # wandb.watch(model, log='all')

        if self.use_transformer:
            opt = torch.optim.Adam(
                list(model.parameters()),
                2e-4,
                # betas=(0.9, 0.98),  # B in Lingvo; in Trfmr paper.
                betas=(0.9, 0.997),  # A in Lingvo.
                eps=1e-9,
            )
        else:
            if self.optimizer == 'adam':
                opt = torch.optim.Adam(list(model.parameters()), 2e-4)
            else:
                print('Using Adagrad')
                opt = torch.optim.Adagrad(list(model.parameters()), 2e-4)
        self.opt = opt
        total_steps = self.epochs * self.max_steps # max_step 과 epoch 수로  total_step 으로 lr scheduler 를 설정한다.
        if self.lr_scheduler == 'CosineAnnealingLR':
            # Starts decaying to 0 immediately.
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,total_steps)#total_steps
        elif self.lr_scheduler == 'OneCycleLR':
            # Warms up to max_lr, then decays to ~0.
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt, max_lr=2e-3, total_steps=total_steps)
        elif self.lr_scheduler is not None and self.lr_scheduler.startswith(
                'OneCycleLR-'):
            warmup_percentage = float(self.lr_scheduler.split('-')[-1])
            # Warms up to max_lr, then decays to ~0.
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=2e-3,
                total_steps=total_steps,
                pct_start=warmup_percentage)
        elif self.lr_scheduler is not None and self.lr_scheduler.startswith(
                'wd_'):
            # Warmups and decays.
            splits = self.lr_scheduler.split('_')
            assert len(splits) == 3, splits
            lr, warmup_fraction = float(splits[1]), float(splits[2])
            self.custom_lr_lambda = train_utils.get_cosine_learning_rate_fn(
                total_steps,
                learning_rate=lr,
                min_learning_rate_mult=1e-5,
                constant_fraction=0.,
                warmup_fraction=warmup_fraction)
        else:
            assert self.lr_scheduler is None, self.lr_scheduler

        self.tbx_logger = tune_logger.TBXLogger(self.config, self.logdir)
        if self.checkpoint_to_load:
            self.LoadCheckpoint()

        self.loaded_queries = None
        self.oracle_cards = None
        # +@ use available_dataset
        not_support_query_idx = list()
        if self.dataset in available_dataset: #and len(self.join_tables) > 1:
            # +@ we change JobToQuery to FormattingQuery
            # queries_job_format = utils.FormattingQuery(self.queries_csv,sep=self.sep)
            queries_job_format, not_support_query_idx = utils.FormattingQuery_JoinFilter(self.queries_csv,self.join_clauses,sep=self.sep)
            self.loaded_queries, self.oracle_cards = utils.UnpackQueries(
                self.table, queries_job_format)
        self.not_support_query_idx = not_support_query_idx
        print(f"Not support queries : {not_support_query_idx}")
        if config['__gpu'] == 0:
            print('CUDA not available, using # cpu cores for intra-op:',
                  torch.get_num_threads(), '; inter-op:',
                  torch.get_num_interop_threads())

        if not self.use_data_parallel:
            print("DO NOT USE MULTI GPU")

        if hasattr(self,'update_dir')  and self.update_dir is not None:
            table.cardinality = self.UpdateSampler(self.join_tables,self.update_dir)
            self.train_data.cardinality = table.cardinality

        if hasattr(self,'query_driven') and self.query_driven: 
            if hasattr(self,'q_run_size'):
                assert self.q_run_size <= self.q_bs
                self.query_run_size = self.q_run_size
            elif hasattr(self,'q_bs'):
                self.query_run_size = self.q_bs
            else: 
                assert False

            self.query_driven_estimator = self.MakeProgressiveSampler_train(
                    self.model,
                    self.train_data if self.factorize else self.table,
                    do_fanout_scaling=(self.dataset in available_dataset) and (len(self.join_tables) > 1),
                    train_virtual_cols=self.train_virtual_cols,
                    batch_size=self.query_run_size
                    )


            if self.mode == 'INFERENCE' or self.eval_join_sampling:
                return
            not_support_query_idx = list()
            queries_job_format, not_support_query_idx = utils.FormattingQuery_JoinFilter(self.train_queries_csv, self.join_clauses, sep=self.sep)
            if len(queries_job_format) != self.train_queries:
                queries_job_format = queries_job_format[:train_queries]
            self.train_loaded_queries, self.train_oracle_cards = utils.UnpackQueries(
                    self.table, queries_job_format)
            assert len(not_support_query_idx) == 0, "TRAIN DATA CONTAINS UNSUPPORTED QUERY"
            print(f'total {len(self.train_loaded_queries)} training queries unpacked')
            assert len(self.train_loaded_queries) == self.train_queries
            #self.query_driven_dataset = common.QueryDrivenDataset(train_loaded_queries, train_oracle_cards, self.query_driven_estimator)
            #if not hasattr(self, 'query_bs'):
            #    self.query_bs = len(queries_job_format) / self.max_steps
            #print(f'query_bs = {self.query_bs}')
            #self.query_driven_dataset.query_bs = int(np.floor(self.query_bs))

            if self.use_query_validation:
                assert hasattr(self,'validation_queries_csv')

                self.validation_estimator = self.MakeProgressiveSampler_train(
                        self.model,
                        self.train_data if self.factorize else self.table,
                        # +@
                        do_fanout_scaling=(self.dataset in available_dataset) and (len(self.join_tables) > 1),
                        train_virtual_cols=self.train_virtual_cols,
                        batch_size=self.query_run_size,
                        is_train=False,
                        )
                queries_job_format, not_support_query_idx = utils.FormattingQuery_JoinFilter(self.validation_queries_csv, self.join_clauses, sep=self.sep)
                self.validation_loaded_queries, self.validation_cards = utils.UnpackQueries(self.table, queries_job_format)
                assert len(not_support_query_idx) == 0, "VALIDATION DATA CONTAINS UNSUPPORTED QUERY"
                print(f'total {len(self.validation_loaded_queries)} validation queries unpacked')

        else:
            self.query_driven = False
            self.query_driven_estimator = None

        if not hasattr(self, 'query_only'):
            self.query_only = False

    def LoadCheckpoint(self):
        all_ckpts = glob.glob(self.checkpoint_to_load)
        msg = f'No ckpt found or use tune.grid_search() for >1 ckpts {all_ckpts} {self.checkpoint_to_load}.'
        assert len(all_ckpts) == 1, msg
        loaded = torch.load(all_ckpts[0])
        # try:
        if isinstance(self.model, DataParallelPassthrough):
            self.model.module.load_state_dict(loaded['model_state_dict'])
        else:
            self.model.load_state_dict(loaded['model_state_dict'])

        #train_utils.ReportModel(self.model)
        
        # +@ expand load ckpt for re-training
        if self.mode == 'INFERENCE' :
            print('Loaded ckpt from', all_ckpts[0])
            return None
        self.epoch = loaded['epoch']
        self.opt.load_state_dict(loaded['optimizer_state_dict'])
        self.lloss = loaded['loss']
        # except RuntimeError as e:
        #     # Backward compatibility: renaming.
        #     def Rename(state_dict):
        #         new_state_dict = collections.OrderedDict()
        #         for key, value in state_dict.items():
        #             new_key = key
        #             if key.startswith('embedding_networks'):
        #                 new_key = key.replace('embedding_networks',
        #                                       'embeddings')
        #             new_state_dict[new_key] = value
        #         return new_state_dict
        #
        #     loaded = Rename(loaded)
        #
        #     modules = list(self.model.net.children())
        #     if len(modules) < 2 or type(modules[-2]) != nn.ReLU:
        #         raise e
        #     # Try to load checkpoints created prior to a 7/28/20 fix where
        #     # there's an activation missing.
        #     print('Try loading without ReLU before output layer.')
        #     modules.pop(-2)
        #     self.model.net = nn.Sequential(*modules)
        #     self.model.load_state_dict(loaded)

        print('Loaded ckpt from', all_ckpts[0])

    def MakeTableDataset(self, table):
        train_data = common.TableDataset(table)
        if self.factorize:
            train_data = common.FactorizedTable(
                train_data, word_size_bits=self.word_size_bits)
        return train_data

    def MakeSamplerDatasetLoader(self, loaded_tables):
        assert self.sampler in ['fair_sampler',
                                'factorized_sampler'], self.sampler
        join_spec = join_utils.get_join_spec(self.__dict__)
        if self.sampler == 'fair_sampler':
            klass = fair_sampler.FairSamplerIterDataset
        else:
            klass = factorized_sampler.FactorizedSamplerIterDataset
        join_iter_dataset = klass(
            loaded_tables,
            join_spec,
            # +@ for load another dataset not only imdb, need these 3 parameters
            data_dir = self.data_dir,
            dataset = self.dataset,
            use_cols = self.use_cols,
            rust_random_seed = self.rust_random_seed,
            rng = self.rng,


            sample_batch_size=self.sampler_batch_size,
            disambiguate_column_names=True,

            # Only initialize the sampler if training.
            # +@ for retraining, have to change setting
            initialize_sampler= True,#(self.mode !='INFERENCE'),
            save_samples=self._save_samples,
            load_samples=self._load_samples)

        table = common.ConcatTables(loaded_tables,
                                    self.join_keys,
                                    sample_from_join_dataset=join_iter_dataset)
        if self.factorize:
            join_iter_dataset = common.FactorizedSampleFromJoinIterDataset(
                join_iter_dataset,
                base_table=table,
                factorize_blacklist=self.dmol_cols if self.num_dmol else
                self.factorize_blacklist if self.factorize_blacklist else [],
                word_size_bits=self.word_size_bits,
                factorize_fanouts=self.factorize_fanouts)


        # +@
        # loader = None
        loader = data.DataLoader(join_iter_dataset,
                                 batch_size=self.bs)
                                 # num_workers=self.loader_workers,
                                 # worker_init_fn=lambda worker_id: np.random.
                                 # seed(np.random.get_state()[1][0] + worker_id),
                                 # pin_memory=True)
        return join_spec, join_iter_dataset, loader, table

    def UpdateSampler(self, join_tables,update_dir):
        loaded_tables = list()
        for t in join_tables:
            table = datasets.LoadDataset(self.dataset,t,data_dir=update_dir, use_cols=self.use_cols)
            table.data.info()
            loaded_tables.append(table)

        cache_df_list = glob.glob('cache/*.df')
        for path in cache_df_list:
            os.remove(path)
        cache_dir = glob.glob(f"cache/{self.join_name}*")[0]
        shutil.rmtree(cache_dir)

        print("Remove cache files")

        assert self.sampler in ['fair_sampler',
                                'factorized_sampler'], self.sampler

        sampler = factorized_sampler.FactorizedSampler(loaded_tables, self.join_spec,
                                         self.sampler_batch_size,
                                         update_dir,
                                         self.dataset,
                                         self.use_cols,
                                         self.rust_random_seed,
                                         self.rng)
        del self.train_data.join_iter_dataset.sampler
        gc.collect()
        self.train_data.join_iter_dataset.sampler = sampler
        self.epoch = 0
        true_card = sampler.join_card
        print(f"True cardinality - from jct  {true_card}")

        return true_card

    def MakeOrdering(self, table):
        fixed_ordering = None
        if self.dataset not in available_dataset and self.special_orders <= 1:
            fixed_ordering = list(range(len(table.columns)))

        if self.order is not None:
            print('Using passed-in order:', self.order)
            fixed_ordering = self.order

        if self.order_seed is not None:
            if self.order_seed == 'reverse':
                fixed_ordering = fixed_ordering[::-1]
            else:
                rng = np.random.RandomState(self.order_seed)
                rng.shuffle(fixed_ordering)
            print('Using generated order:', fixed_ordering)
        return fixed_ordering

    def MakeModel(self, table, train_data, table_primary_index=None):
        cols_to_train = table.columns
        if self.factorize:
            cols_to_train = train_data.columns

        fixed_ordering = self.MakeOrdering(table)

        table_num_columns = table_column_types = table_indexes = None
        if isinstance(train_data, (common.SamplerBasedIterDataset,
                                   common.FactorizedSampleFromJoinIterDataset)):
            table_num_columns = train_data.table_num_columns
            table_column_types = train_data.combined_columns_types
            table_indexes = train_data.table_indexes
            print('table_num_columns', table_num_columns)
            print('table_column_types', table_column_types)
            print('table_indexes', table_indexes)
            print('table_primary_index', table_primary_index)

        if self.use_transformer:
            args = {
                'num_blocks': 4,
                'd_ff': 128,
                'd_model': 32,
                'num_heads': 4,
                'd_ff': 64,
                'd_model': 16,
                'num_heads': 2,
                'nin': len(cols_to_train),
                'input_bins': [c.distribution_size for c in cols_to_train],
                'use_positional_embs': False,
                'activation': 'gelu',
                'fixed_ordering': self.fixed_ordering,
                'dropout': self.dropout,
                'per_row_dropout': self.per_row_dropout,
                'seed': None,
                'join_args': {
                    'num_joined_tables': len(self.join_tables),
                    'table_dropout': self.table_dropout,
                    'table_num_columns': table_num_columns,
                    'table_column_types': table_column_types,
                    'table_indexes': table_indexes,
                    'table_primary_index': table_primary_index,
                }
            }
            args.update(self.transformer_args)
            model = transformer.Transformer(**args).to(train_utils.get_device())
        else:
            model = MakeMade(
                table=table,
                scale=self.fc_hiddens,
                layers=self.layers,
                cols_to_train=cols_to_train,
                seed=self.seed,
                factor_table=train_data if self.factorize else None,
                fixed_ordering=fixed_ordering,
                special_orders=self.special_orders,
                order_content_only=self.order_content_only,
                order_indicators_at_front=self.order_indicators_at_front,
                inv_order=True,
                residual=self.residual,
                direct_io=self.direct_io,
                input_encoding=self.input_encoding,
                output_encoding=self.output_encoding,
                embed_size=self.embed_size,
                dropout=self.dropout,
                per_row_dropout=self.per_row_dropout,
                grouped_dropout=self.grouped_dropout
                if self.factorize else False,
                fixed_dropout_ratio=self.fixed_dropout_ratio,
                input_no_emb_if_leq=self.input_no_emb_if_leq,
                embs_tied=self.embs_tied,
                resmade_drop_prob=self.resmade_drop_prob,
                # DMoL:
                num_dmol=self.num_dmol,
                scale_input=self.scale_input if self.num_dmol else False,
                dmol_cols=self.dmol_cols if self.num_dmol else [],
                # Join specific:
                num_joined_tables=len(self.join_tables),
                table_dropout=self.table_dropout,
                table_num_columns=table_num_columns,
                table_column_types=table_column_types,
                table_indexes=table_indexes,
                table_primary_index=table_primary_index,
            )
        return model

    def MakeProgressiveSamplers(self,
                                model,
                                train_data,
                                do_fanout_scaling=False,
                                sample_size=None
                                ):
        sizes=self.eval_psamples
        estimators = []
        dropout = self.dropout or self.per_row_dropout
        if sample_size is not None:
            #sizes.insert(0, sample_size)
            sizes = [sample_size]
        assert len(sizes) == 1
        for n in sizes:
            if self.factorize:
                estimators.append(
                    estimators_lib.FactorizedProgressiveSampling(
                        model,
                        train_data,
                        n,
                        self.join_spec,
                        device=train_utils.get_device(),
                        shortcircuit=dropout,
                        do_fanout_scaling=do_fanout_scaling))
            else:
                estimators.append(
                    estimators_lib.ProgressiveSampling(
                        model,
                        train_data,
                        n,
                        self.join_spec,
                        device=train_utils.get_device(),
                        shortcircuit=dropout,
                        do_fanout_scaling=do_fanout_scaling))
        # +@ for save verbose
        if self.verbose_mode:
            for est in estimators :
                est.set_verbose_data(len(self.loaded_queries))
        return estimators

    def MakeProgressiveSampler_train(self,
                                    model,
                                    train_data,
                                    do_fanout_scaling=False,
                                    train_virtual_cols=True,
                                    batch_size=None,
                                    is_train=True
                                    ):
        dropout = self.dropout or self.per_row_dropout

        if batch_size is None:
            batch_size = math.ceil(self.train_queries / self.max_steps)
            batch_size = int(batch_size)
        if self.factorize:
            res = estimators_lib.BatchDifferentiableFactorizedProgressiveSampling(
                    model,
                    train_data,
                    self.train_sample_num,
                    self.join_spec,
                    device=train_utils.get_device(),
                    shortcircuit=dropout,
                    do_fanout_scaling=do_fanout_scaling,
                    train_virtual_cols=train_virtual_cols,
                    tau=self.gumbel_tmp,
                    batch_size=batch_size,
                    is_training=is_train)
        else:
            res = estimators_lib.BatchDifferentiableProgressiveSampling(
                    model,
                    train_data,
                    self.train_sample_num,
                    self.join_spec,
                    device=train_utils.get_device(),
                    shortcircuit=dropout,
                    do_fanout_scaling=do_fanout_scaling,
                    train_virtual_cols=train_virtual_cols,
                    tau=self.gumbel_tmp,
                    batch_size=batch_size,
                    is_training=is_train)
        return res

    def _simple_save(self):


        workload = self.workload_name
        os.makedirs(MODEL_PATH,exist_ok=True)
        PATH = f'{MODEL_PATH}/{workload}_{self.epochs}_{self.q_weight}_{self.gumbel_tmp}.tar'
        print('Saved to:', PATH)
        # wandb.save(PATH)

        if isinstance(self.model, DataParallelPassthrough):
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        save_state = {
            'epoch': self.epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.opt.state_dict(),
            'lr_state_dict':self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'loss': self.lloss,
            'train_time' : self.total_train_time if self.mode == "TRAIN" else 0,

        }
        torch.save(save_state,PATH)

        return PATH

    def _train(self):
        # +@ check inference mode
        if self.mode == 'INFERENCE' or self.eval_join_sampling:
            self.model.model_bits = 0 
            results = self.evaluate(self.num_eval_queries_at_checkpoint_load,
                                    done=True)
            self._maybe_check_asserts(results, returns=None)
            return {
                'epoch': 0,
                'done': True,
                'results': results,
            }

        t1 = time.time()
        for _ in range(min(self.epochs - self.epoch, 
                           self.epochs_per_iteration)):  
            mean_epoch_train_loss = run_epoch( 
                'train',
                self.model,
                self.opt,
                upto=self.max_steps if self.dataset in available_dataset else None,
                train_data=self.train_data,
                val_data=self.train_data,
                batch_size=self.bs,
                epoch_num=self.epoch,
                epochs=self.epochs,
                log_every=500,
                table_bits=self.table_bits,
                warmups=self.warmups,
                loader=self.loader,
                query_driven=self.query_driven,
                query_only=self.query_only,
                query_driven_estimator=self.query_driven_estimator,
                query_list=self.train_loaded_queries if hasattr(self,'train_loaded_queries') else None ,
                card_list=self.train_oracle_cards  if hasattr(self,'train_oracle_cards') else None ,
                q_bs=self.q_bs if hasattr(self, 'q_bs') else None,
                q_weight=self.q_weight,
                constant_lr=self.constant_lr,
                summary_writer=self.tbx_logger._file_writer,
                lr_scheduler=self.lr_scheduler,
                custom_lr_lambda=self.custom_lr_lambda,
                label_smoothing=self.label_smoothing,
                # +@ pass instance
                neurocard_instance=self,
                sep_backward=self.sep_backward)
            self.epoch += 1
        self.model.model_bits = mean_epoch_train_loss / np.log(2) 

        # +@ log total train time

        total_train_time = time.time() - t1
        if self.mode == 'TRAIN':
            self.total_train_time += total_train_time

        if self.use_query_validation:
            print("Model validation")
            val_loss = self.query_validation()
            self.val_loss = val_loss.detach().cpu().item()
        else:
            self.val_loss = 0

        done = self.epoch >= self.epochs
        results = self.evaluate( 
            max(self.num_eval_queries_at_end, 
                self.num_eval_queries_per_iteration)
            if done else self.num_eval_queries_per_iteration, done)

        returns = {
            'epochs': self.epoch,
            'done': done,
            'avg_loss': self.model.model_bits - self.table_bits,
            'train_bits': self.model.model_bits,
            'train_bit_gap': self.model.model_bits - self.table_bits,
            'results': results,
            'total_train_time': self.total_train_time,
            'bs' : self.bs,
            'q_loss': self.q_loss,
            'd_loss': self.d_loss,
            'val_loss': self.val_loss,
            'q_weight' : self.q_weight,
            'gumbel_tmp' : self.gumbel_tmp,
        }

        if self.compute_test_loss:
            returns['test_bits'] = np.mean(
                run_epoch(
                    'test',
                    self.model,
                    opt=None,
                    train_data=self.train_data,
                    val_data=self.train_data,
                    batch_size=self.bs,
                    upto=None if self.dataset not in available_dataset else 20,
                    log_every=500,
                    table_bits=self.table_bits,
                    return_losses=True
                    )) / np.log(2)
            self.model.model_bits = returns['test_bits']
            print('Test bits:', returns['test_bits'])

        if done:
            self._maybe_check_asserts(results, returns)

        now = datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
        loss_path = f"./loss_results/{self.workload_name}_{args.loss_file}"
        with open(loss_path,'at') as writer:
            txt = f"{self.workload_name},{self.epoch},{self.epochs},{self.model.model_bits - self.table_bits},{self.val_loss},{self.d_loss},{self.q_loss},{self.total_train_time},{self.q_weight},{self.gumbel_tmp},{self.bs},{now}\n"
            writer.write(txt)
        auto_garbage_collect()

        if self.mode == 'TRAIN' and self.checkpoint_every_epoch:
            self.avg_loss = self.model.model_bits - self.table_bits
            if self.mininum_loss == -1 :
                self.mininum_loss = self.avg_loss
            if self.avg_loss < self.mininum_loss:
                self.mininum_loss = self.avg_loss
                self._simple_save()

        # if :
        return returns

    def query_validation(self):
        query_card_list = list(zip(self.validation_loaded_queries, self.validation_cards))



        val_queries = [q for q, c in query_card_list]
        val_cards = [c for q, c in query_card_list]
        val_cards = np.array(val_cards)
        val_cards = torch.as_tensor(val_cards, dtype=torch.float32)
        val_cards = val_cards.to(train_utils.get_device())
        cols_list = [query[0] for i, query in enumerate(val_queries)]
        ops_list = [query[1] for i, query in enumerate(val_queries)]
        vals_list = [query[2] for i, query in enumerate(val_queries)]

        num_query = len(self.validation_loaded_queries)
        query_bs = self.query_run_size
        est_card_batch_list = list()

        num_iter = num_query//query_bs
        cur_idx = 0
        for i in range(num_iter):
            est_card_batch = self.validation_estimator.Query(cols_list[cur_idx:cur_idx+query_bs], ops_list[cur_idx:cur_idx+query_bs], vals_list[cur_idx:cur_idx+query_bs]).detach().cpu()
            cur_idx += query_bs
            est_card_batch_list.append(est_card_batch)

        if cur_idx < num_query:
            est_card_batch = self.validation_estimator.Query(cols_list[cur_idx:], ops_list[cur_idx:], vals_list[cur_idx:]).detach().cpu()
            est_card_batch_list.append(est_card_batch)


        est_card_batch = torch.cat(est_card_batch_list)

        val_loss = get_qerror_torch(est_card_batch.to(train_utils.get_device()), val_cards)
        print('validation loss',':',val_loss)
        return val_loss


    def _maybe_check_asserts(self, results, returns): 
        if self.asserts:
            # asserts = {key: val, ...} where key either exists in "results"
            # (returned by evaluate()) or "returns", both defined above.
            error = False
            message = []
            for key, max_val in self.asserts.items():  
                if key in results:
                    if results[key] >= max_val: 
                        error = True
                        message.append(str((key, results[key], max_val))) 
                elif returns[key] >= max_val: 
                    error = True
                    message.append(str((key, returns[key], max_val)))
            assert not error, '\n'.join(message)

    def _save(self, tmp_checkpoint_dir):

        rep_path = f"../../models/UAE/{self.workload_name}.tar"

        if self.mode == "TRAIN":
            if isinstance(self.model, DataParallelPassthrough):
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
            save_state ={
                'epoch': self.epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.opt.state_dict(),
                'lr_state_dict':self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                'loss': self.lloss,
                'train_time' : self.total_train_time,
            }
            os.makedirs(f"../../models/UAE/",exist_ok=True)
            try:
                torch.save(save_state, rep_path)
            except: # if no disk
                rep_path = f'{MODEL_PATH}/{self.workload_name}.tar'
                torch.save(save_state, rep_path)

        return {'path': rep_path}

    def stop(self):
        self.tbx_logger.flush()
        self.tbx_logger.close()

    def _log_result(self, results):
        psamples = {}
        # When we run > 1 epoch in one tune "iter", we want TensorBoard x-axis
        # to show our real epoch numbers.
        results['iterations_since_restore'] = results[
            'training_iteration'] = self.epoch
        for k, v in results['results'].items():
            if 'psample' in k:
                psamples[k] = v
        # wandb.log(results)
        self.tbx_logger.on_result(results)
        self.tbx_logger._file_writer.add_custom_scalars_multilinechart(
            map(lambda s: 'ray/tune/results/{}'.format(s), psamples.keys()),
            title='psample')

    def ErrorMetric(self, est_card, card):
        if card == 0 and est_card != 0:
            return est_card
        if card != 0 and est_card == 0:
            return card
        if card == 0 and est_card == 0:
            return 1.0
        return max(est_card / card, card / est_card)

    def Query(self,
              estimators,
              oracle_card=None,
              query=None,
              table=None,
              oracle_est=None):
        assert query is not None
        cols, ops, vals = query
        card = oracle_est.Query(cols, ops,
                                vals) if oracle_card is None else oracle_card
        print('Q(', end='')
        for c, o, v in zip(cols, ops, vals):
            print('{} {} {}, '.format(c.name, o, str(v)), end='')
        print('): ', end='')
        print('\n  actual {} ({:.3f}%) '.format(card,
                                                card / table.cardinality * 100),
              end='')
        # +@
        for est in estimators:
            #est_card = est.Query(cols, ops, vals, query_driven=self.query_driven)
            est_card = est.Query(cols, ops, vals)
            err = self.ErrorMetric(est_card, card)
            est.AddError(err, est_card, card)

            print('{} {} (err={:.3f}) '.format(str(est), est_card, err), end='')
        print()

    def NotSupportQuery(self,estimators):
        for est in estimators:
            est_card = est.Query([], [], [],not_support=True)
            err = -1
            est.AddError(-1, est_card, -1)
            print('Not supported query ')


    def evaluate(self, num_queries, done, estimators=None):
        model = self.model
        if isinstance(model, DataParallelPassthrough):
            model = model.module
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        results = {}
        if num_queries:
            if estimators is None:
                estimators = self.MakeProgressiveSamplers(
                    model,
                    self.train_data if self.factorize else self.table,
                    # +@
                    do_fanout_scaling=(self.dataset in available_dataset) and (len(self.join_tables) > 1) )
                if self.eval_join_sampling:  # None or an int.
                    estimators = [
                        estimators_lib.JoinSampling(self.train_data, self.table,
                                                    self.eval_join_sampling)
                    ]

            assert self.loaded_queries is not None
            num_queries = min(len(self.loaded_queries), num_queries)
            err_queries = list()
            for i in range(num_queries):
                gc.collect()
                torch.cuda.empty_cache()
                print('Query {}:'.format(i), end=' ')
                query = self.loaded_queries[i]
                if i in self.not_support_query_idx:
                    self.NotSupportQuery(estimators)
                else:
                    try:
                        self.Query(estimators,
                                   oracle_card=None if self.oracle_cards is None else
                                   self.oracle_cards[i],
                                   query=query,
                                   table=self.table,
                                   oracle_est=self.oracle)
                    except Exception as e:
                        err_queries.append(i)
                        trace_log = traceback.format_exc().replace('\n','  ')
                        err_line = f"err - {i}\t{trace_log}\n"
                        with open(f'{self.workload_name}_err_query_trace.log','at') as writer:
                            writer.write(err_line)
                        gc.collect()
                        torch.cuda.empty_cache()

                print(f'{i+1}/{num_queries} report')
                if i % 100 == 0:
                    for est in estimators:
                        est.report()
            print(f"inf err queries {err_queries}")
            if len(err_queries) > 0:
                with open(f'{self.workload_name}_err_queries.txt','at') as writer:
                    writer.write(f"err queries {err_queries}\n")
            # +@ add enumerate for save result
            for i,est in enumerate(estimators):
                results[str(est) + '_max'] = np.max(est.errs)
                results[str(est) + '_p99'] = np.quantile(est.errs, 0.99)
                results[str(est) + '_p95'] = np.quantile(est.errs, 0.95)
                results[str(est) + '_median'] = np.median(est.errs)
                est.report()

                series = pd.Series(est.query_dur_ms)
                print(series.describe())
                series.to_csv(str(est) + '.csv', index=False, header=False)

                # +@ save result to csv
                if self.verbose_mode:
                    est.save_verbose_data(self.workload_name)
                if self.save_eval_result :
                    workload = self.workload_name
                    out_dir = f'../../results/UAE'
                    if not os.path.isdir(out_dir):
                        os.mkdir(out_dir)

                    now = datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S')

                    err_df = pd.DataFrame([est.errs,est.est_cards,est.true_cards,est.query_dur_ms]).transpose()
                    err_df.to_csv(f'{out_dir}/{workload}.csv',index=False,header=['errs','est_cards','true_cards','query_dur_ms'])

        return results


if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)
    MODEL_PATH = '/mnt/disk2/models'
    # ray.init(address= "auto", ignore_reinit_error=True)


    for k in args.run:
        assert k in experiments.EXPERIMENT_CONFIGS, 'Available: {}'.format(
            list(experiments.EXPERIMENT_CONFIGS.keys()))

    num_gpus = args.gpus if torch.cuda.is_available() else 0
    num_cpus = args.cpus



    train_tuple_dir = './sample_tuples'
    if not os.path.isdir(train_tuple_dir):
        os.mkdir(train_tuple_dir)
    result_dir = './results'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    loss_result_dir = './loss_results'
    os.makedirs(loss_result_dir, exist_ok=True)

    eval_result_dir = f'../../results/UAE'
    os.makedirs(eval_result_dir, exist_ok=True)
        

    workload_name = args.run[0]

    tune_result = f'./tune_results/{workload_name}'

    os.makedirs(tune_result, exist_ok=True)

    log_path = f"{result_dir}/log.txt"
    now = datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S:%f')
    with open(log_path,'at') as writer : writer.write(f'{now}\n')

    # worker_ip = requests.get("http://ip.jsontest.com").json()['ip']


    config = dict(experiments.EXPERIMENT_CONFIGS[workload_name],
                  **{    '__run': workload_name,
                         'workload_name': workload_name,
                         '__gpu': num_gpus,
                         '__cpu': num_cpus,})

    if args.tuning:
        asha_scheduler = ASHAScheduler(max_t=config['epochs']+1,grace_period=10,reduction_factor=2,metric='avg_loss',mode='min')
        analysis =tune.run(NeuroCard,name="neurocard",scheduler=asha_scheduler,num_samples=1,
                 resources_per_trial={ "cpu":1, "gpu":1}, config=config,local_dir=tune_result)
    else:

        tune.run_experiments(
        {
            k: {
                'run': NeuroCard,
                'checkpoint_at_end': True,
                'resources_per_trial': {
                    'gpu': num_gpus,
                    'cpu': num_cpus,
                },
                'config': dict(
                    experiments.EXPERIMENT_CONFIGS[k], **{
                        'workload_name' : k,
                        '__run': k,
                        '__gpu': num_gpus,
                        '__cpu': num_cpus
                    }),
            } for k in args.run
        },
        concurrent=True,
        )


