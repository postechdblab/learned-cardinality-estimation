"""Evaluate estimators (Naru or others) on queries."""
import argparse
import collections
import glob
import os
import pickle
import re
import time

import numpy as np
import pandas as pd
import torch

import naru.common as common 
from naru.common import *
#import naru.common.config as config
import naru.datasets as datasets
import naru.estimators as estimators_lib
import naru.made as made
import naru.transformer as transformer

# For inference speed.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def MakeTable():
    dataset = config['dataset']
    assert dataset in ['dmv-tiny', 'dmv', 'custom']
    if dataset == 'dmv-tiny':
        table = datasets.LoadDmv('dmv-tiny.csv')
    elif dataset == 'dmv':
        table = datasets.LoadDmv()
    elif dataset == 'custom':
        table = datasets.LoadCustom()

    oracle_est = estimators_lib.Oracle(table)
    if config['run-bn']:
        return table, common.TableDataset(table), oracle_est
    return table, None, oracle_est


def ErrorMetric(est_card, card):
    if card == 0 and est_card != 0:
        return est_card
    if card != 0 and est_card == 0:
        return card
    if card == 0 and est_card == 0:
        return 1.0
    return max(est_card / card, card / est_card)


def SampleTupleThenRandom(all_cols,
                          num_filters,
                          rng,
                          table,
                          return_col_idx=False):
    s = table.data.iloc[rng.randint(0, table.cardinality)]
    vals = s.values

    #if args.dataset in ['dmv', 'dmv-tiny']:
    #    # Giant hack for DMV.
    #    vals[6] = vals[6].to_datetime64()

    idxs = rng.choice(len(all_cols), replace=True, size=num_filters)
    cols = np.take(all_cols, idxs)

    # If dom size >= 10, okay to place a range filter.
    # Otherwise, low domain size columns should be queried with equality.
    ops = rng.choice(['<=', '>=', '='], size=num_filters)
    ops_all_eqs = ['='] * num_filters
    sensible_to_do_range = [c.DistributionSize() >= 10 for c in cols]
    ops = np.where(sensible_to_do_range, ops, ops_all_eqs)

    if num_filters == len(all_cols):
        if return_col_idx:
            return np.arange(len(all_cols)), ops, vals
        return all_cols, ops, vals

    vals = vals[idxs]
    if return_col_idx:
        return idxs, ops, vals

    return cols, ops, vals


def GenerateQuery(all_cols, rng, table, return_col_idx=False):
    """Generate a random query."""
    num_filters = rng.randint(5, 12)
    cols, ops, vals = SampleTupleThenRandom(all_cols,
                                            num_filters,
                                            rng,
                                            table,
                                            return_col_idx=return_col_idx)
    return cols, ops, vals


def Query(estimators,
          do_print=True,
          oracle_card=None,
          query=None,
          table=None,
          oracle_est=None):
    assert query is not None
    cols, ops, vals = query

    ### Actually estimate the query.

    pprint('Q(', end='')
    for c, o, v in zip(cols, ops, vals):
        pprint('{} {} {}, '.format(c.name, o, str(v)), end='')
    pprint('): ', end='')

    for est in estimators:
        est_card = est.Query(cols, ops, vals)
        pprint('{} {}'.format(str(est), est_card), end='')
    pprint()

def ReportEsts(estimators):
    v = -1
    for est in estimators:
        print(est.name, 'max', np.max(est.errs), '99th',
              np.quantile(est.errs, 0.99), '95th', np.quantile(est.errs, 0.95),
              'median', np.quantile(est.errs, 0.5))
        v = max(v, np.max(est.errs))
    return v


def RunN(table,
         cols,
         estimators,
         rng=None,
         num=20,
         log_every=50,
         num_filters=11,
         oracle_cards=None,
         oracle_est=None):
    if rng is None:
        rng = np.random.RandomState(1234)

    last_time = None
    for i in range(num):
        do_print = False
        if i % log_every == 0:
            if last_time is not None:
                print('{:.1f} queries/sec'.format(log_every /
                                                  (time.time() - last_time)))
            do_print = True
            print('Query {}:'.format(i), end=' ')
            last_time = time.time()
        query = GenerateQuery(cols, rng, table)
        Query(estimators,
              do_print,
              oracle_card=None,
              query=query,
              table=table,
              oracle_est=oracle_est)

    return False


def RunNParallel(estimator_factory,
                 parallelism=2,
                 rng=None):
    """RunN in parallel with Ray.  Useful for slow estimators e.g., BN."""
    import ray
    ray.init(redis_password='xxx')

    @ray.remote
    class Worker(object):

        def __init__(self, i):
            self.estimators, self.table, self.oracle_est = estimator_factory()
            self.columns = np.asarray(self.table.columns)
            self.i = i

        def run_query(self, query, j):
            col_idxs, ops, vals = pickle.loads(query)
            Query(self.estimators,
                  do_print=True,
                  oracle_card=oracle_cards[j]
                  if oracle_cards is not None else None,
                  query=(self.columns[col_idxs], ops, vals),
                  table=self.table,
                  oracle_est=self.oracle_est)

            print('=== Worker {}, Query {} ==='.format(self.i, j))
            for est in self.estimators:
                est.report()

        def get_stats(self):
            return [e.get_stats() for e in self.estimators]

    print('Building estimators on {} workers'.format(parallelism))
    workers = []
    for i in range(parallelism):
        workers.append(Worker.remote(i))

    print('Building estimators on driver')
    estimators, table, _ = estimator_factory()
    cols = table.columns

    if rng is None:
        rng = np.random.RandomState(1234)
    queries = []
    for i in range(num):
        col_idxs, ops, vals = GenerateQuery(cols,
                                            rng,
                                            table=table,
                                            return_col_idx=True)
        queries.append((col_idxs, ops, vals))

    cnts = 0
    for i in range(num):
        query = queries[i]
        print('Queueing execution of query', i)
        workers[i % parallelism].run_query.remote(pickle.dumps(query), i)

    print('Waiting for queries to finish')
    stats = ray.get([w.get_stats.remote() for w in workers])

    print('Merging and printing final results')
    for stat_set in stats:
        for e, s in zip(estimators, stat_set):
            e.merge_stats(s)
    time.sleep(1)

    print('=== Merged stats ===')
    for est in estimators:
        est.report()
    return estimators


def MakeBnEstimators():
    table, train_data, oracle_est = MakeTable()
    estimators = [
        estimators_lib.BayesianNetwork(train_data,
                                       config['bn-samples'],
                                       'chow-liu',
                                       topological_sampling_order=True,
                                       root=config['bn-root'],
                                       max_parents=2,
                                       use_pgm=False,
                                       discretize=100,
                                       discretize_method='equal_freq')
    ]

    for est in estimators:
        est.name = str(est)
    return estimators, table, oracle_est


def MakeMade(model_id, scale, cols_to_train, seed, fixed_ordering=None):
    if config['inv-order']:
        print('Inverting order!')
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        model_id=model_id,
        nin=len(cols_to_train),
        hidden_sizes=[scale] *
        config['layers'] if config['layers'] > 0 else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=config['input-encoding'],
        output_encoding=config['output-encoding'],
        embed_size=32,
        seed=seed,
        do_direct_io_connections=config['direct-io'],
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=config['residual'],
        fixed_ordering=fixed_ordering,
        column_masking=config['column-masking'],
    ).to(DEVICE)

    return model


def MakeTransformer(cols_to_train, fixed_ordering, seed=None):
    return transformer.Transformer(
        num_blocks=config['blocks'],
        d_model=config['dmodel'],
        d_ff=config['dff'],
        num_heads=config['heads'],
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=True,
        activation=config['transformer-act'],
        fixed_ordering=fixed_ordering,
        column_masking=config['column-masking'],
        seed=seed,
    ).to(DEVICE)


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('{} Number of model parameters: {} (~= {:.1f}MB)'.format(model.model_id, num_params, mb))
    print(model)
    return mb


def SaveEstimators(path, estimators, return_df=False):
    # name, query_dur_ms, errs, est_cards, true_cards
    results = pd.DataFrame()
    for est in estimators:
        data = {
            'est': [est.name] * len(est.errs),
            'err': est.errs,
            'est_card': est.est_cards,
            'true_card': est.true_cards,
            'query_dur_ms': est.query_dur_ms,
        }
        results = results.append(pd.DataFrame(data))
    if return_df:
        return results
    results.to_csv(path, index=False)


def LoadOracleCardinalities():
    ORACLE_CARD_FILES = {
        'dmv': 'datasets/dmv-2000queries-oracle-cards-seed1234.csv'
    }
    path = ORACLE_CARD_FILES.get(config['dataset'], None)
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        assert len(df) == 2000, len(df)
        return df.values.reshape(-1)
    return None


def GetNARU(model_id, table_path, model_path):
    selected_ckpts = [model_path]

    print(f'{model_id} ckpts', selected_ckpts)
    print(f'{model_id} Free GPU memory before GetNARU')
    get_gpu_memory()

    table = datasets.LoadCustom(table_path)
    cols_to_train = table.columns
    models = []

    for s in selected_ckpts:
        seed = config['seed']
        order = None

        if config['heads'] > 0:
            model = MakeTransformer(cols_to_train=table.columns,
                                    fixed_ordering=order,
                                    seed=seed)
        else:
            if config['dataset'] in ['dmv-tiny', 'dmv', 'custom']:
                model = MakeMade(
                    model_id=model_id,
                    scale=config['fc-hiddens'],
                    cols_to_train=table.columns,
                    seed=seed,
                    fixed_ordering=order
                )
            else:
                assert False, config['dataset']

        assert order is None or len(order) == model.nin, order
        ReportModel(model)
        print(f'{model_id}, Loading ckpt:', s)
        model.load_state_dict(torch.load(s))
        model.eval()

        models.append(model)

    # Estimators to run.
    estimators = [
        estimators_lib.ProgressiveSampling(m.model_id, m,
                                           table,
                                           config['psample'],
                                           device=DEVICE,
                                           shortcircuit=config['column-masking'],
                                           )
        for m in models 
    ]
    print(f'{model_id} Free GPU memory after GetNARU')
    get_gpu_memory()

    return estimators[0]
