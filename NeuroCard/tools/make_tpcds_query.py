"""Generates new queries on the JOB-light schema.

For each JOB-light join template, repeat #queries per template:
   - sample a tuple from this inner join result via factorized_sampler
   - sample #filters, and columns to put these filters on
   - query literals: use this tuple's values
   - sample ops: {>=, <=, =} for numerical columns and = for categoricals.

Uses either Spark or Postgres for actual execution to obtain true
cardinalities.  Both setups require data already be loaded.  For Postgres, the
server needs to be live.

Typical usage:

To generate queries:
    python make_job_queries.py --output_csv <csv> --num_queries <n>

To print selectivities of already generated queries:
    python make_job_queries.py \
      --print_sel --output_csv queries/job-light.csv --num_queries 70
"""

import os
import subprocess
import textwrap
import time
import shutil

from absl import app
from absl import flags
from absl import logging
from mako import template
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import argparse
import common
import datasets
from factorized_sampler import FactorizedSamplerIterDataset
import join_utils

import psycopg2
import datetime
import time
from glob import glob
import utils

TPCDS_ALIAS_TO_TABLE = {
        'ss': 'store_sales',
        'sr': 'store_returns',
        'cs': 'catalog_sales',
        'cr': 'catalog_returns',
        'ws': 'web_sales',
        'wr': 'web_returns',
        'inv': 'inventory',
        's': 'store',
        'cc': 'call_center',
        'cp': 'catalog_page',
        'web': 'web_site',
        'wp': 'web_page',
        'w': 'warehouse',
        'c': 'customer',
        'ca': 'customer_address',
        'cd': 'customer_demographics',
        'd': 'date_dim',
        'hd': 'household_demographics',
        'i': 'item',
        'ib': 'income_band',
        'p': 'promotion',
        'r': 'reason',
        'sm': 'ship_mode',
        't': 'time_dim',
    }
TPCDS_TABLE_TO_ALIAS = {
        'store_sales':'ss',
        'store_returns':'sr',
        'catalog_sales':'cs',
        'catalog_returns':'cr',
        'web_sales':'ws',
        'web_returns':'wr',
        'inventory':'inv',
        'store':'s',
        'call_center':'cc',
        'catalog_page':'cp',
        'web_site':'web',
        'web_page':'wp',
        'warehouse':'w',
        'customer':'c',
        'customer_address':'ca',
        'customer_demographics':'cd',
        'date_dim':'d',
        'household_demographics':'hd',
        'item':'i',
        'income_band':'ib',
        'promotion':'p',
        'reason':'r',
        'ship_mode':'sm',
        'time_dim':'t',
    }
def get_join_spec_from_file(csvfile, root,use_alias=True):
    join_specs = []
    join_name = csvfile.split('/')[-1].split('.')[0]
    with open(csvfile, 'r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            tables = []
            join_keys = dict()
            join_clauses = line.replace('\n', '').split("#")[1].split(',')
            no_alias_join_clauses = list()
            for join_clause in join_clauses:
                assert len(join_clause.split('=')) == 2, join_clause
                lhs, rhs = join_clause.split('=')

                ltable, lkey = lhs.split('.')
                rtable, rkey = rhs.split('.')
                if use_alias :
                    ltable = TPCDS_ALIAS_TO_TABLE[ltable]
                    rtable = TPCDS_ALIAS_TO_TABLE[rtable]

                if ltable not in tables:
                    tables.append(ltable)
                if rtable not in tables:
                    tables.append(rtable)

                if ltable in join_keys.keys():
                    if lkey not in join_keys[ltable]:
                        join_keys[ltable].append(lkey)
                else:
                    join_keys[ltable] = [lkey]
                if rtable in join_keys.keys():
                    if rkey not in join_keys[rtable]:
                        join_keys[rtable].append(rkey)
                else:
                    join_keys[rtable] = [rkey]
                no_alias_join_clauses.append(f"{ltable}.{lkey}={rtable}.{rkey}")
            tables = list(join_keys.keys())
            # print(f'\n\n\n num {i},{root}\t{tables}\t{join_keys}\t{join_clauses}')
            join_spec = join_utils.get_join_spec({
                "join_tables": tables,
                "join_keys": join_keys,
                "join_root": root,
                "join_clauses": no_alias_join_clauses,
                "join_how": "inner",
                "join_name": f"{join_name}"
                })

            join_specs.append(join_spec)
    return join_specs


def MakeQueries(join_spec, tables_in_templates, use_cols, num_queries, rng, output_file,sep='|'):


    range_workload_cols = []
    light_workload_cols = []
    categoricals = []
    numericals = []

    for table_name in join_spec.join_tables:
        categorical_cols = datasets.TPCDSBenchmark.CATEGORICAL_COLUMNS[
            table_name]
        for c in categorical_cols:
            disambiguated_name = common.JoinTableAndColumnNames(table_name,
                                                                c,
                                                                sep='.')
            range_workload_cols.append(disambiguated_name)
            light_workload_cols.append(disambiguated_name)
            categoricals.append(disambiguated_name)


        range_cols = datasets.TPCDSBenchmark.RANGE_COLUMNS[table_name]
        for c in range_cols:
            disambiguated_name = common.JoinTableAndColumnNames(table_name,
                                                                    c,
                                                                    sep='.')
            range_workload_cols.append(disambiguated_name)
            numericals.append(disambiguated_name)

    join_keys_list = []
    for table_name in join_spec.join_tables:

        for key in join_spec.join_keys[table_name]:
            join_keys_list.append(key)

    ds = FactorizedSamplerIterDataset(tables_in_templates,
                                      join_spec,
                                      data_dir='datasets/tpcds_2_13_0/',
                                      dataset='tpcds',
                                      use_cols=use_cols,
                                      sample_batch_size=512,
                                      disambiguate_column_names=False,
                                      add_full_join_indicators=False,
                                      add_full_join_fanouts=False,
                                      rust_random_seed=1234,
                                      rng=rng)

    tables= [ f'{table} {TPCDS_TABLE_TO_ALIAS[table]}' for table in join_spec.join_tables]
    table_string = ','.join(tables)
    def alias_join_clause(txt,TPCDS_TABLE_TO_ALIAS):
        token = txt.split('=')
        assert len(token)==2
        t1,k1 = token[0].split('.')
        t2,k2 = token[1].split('.')
        t1 = TPCDS_TABLE_TO_ALIAS[t1]
        t2 = TPCDS_TABLE_TO_ALIAS[t2]
        return f"{t1}.{k1}={t2}.{k2}"
    joins = [alias_join_clause(join_c,TPCDS_TABLE_TO_ALIAS) for join_c in join_spec.join_clauses]
    print(joins)
    join_string = ','.join(joins)
    predicate_list,predicate_strings = get_predicates(ds, range_workload_cols, rng, categoricals, num_queries, TPCDS_TABLE_TO_ALIAS)

    txt = ''
    for i in range(num_queries):
        txt += f"{table_string}{sep}{join_string}{sep}{predicate_strings[i]}{sep}_\n"
    print(txt)
    with open(output_file,'at') as writer:
        writer.write(txt)
    return None


def get_predicates(ds,content_cols,rng,categoricals, num_queries, TPCDS_TABLE_TO_ALIAS) :

    ncols = len(content_cols)
    predicate_list = list()
    predicate_strings = []

    sampled_df = ds.sampler.run()[content_cols]


    for r in sampled_df.iterrows():
        if len(predicate_strings) == num_queries:
            break
        tup = r[1]
        try :
            num_filters = rng.randint( min(ncols,2) , max(( ncols //2)+2, ncols))
        except :
            num_filters = ncols
        # Positions where the values are non-null.
        non_null_indices = np.argwhere(~pd.isnull(tup).values).reshape(-1, )

        if len(non_null_indices) < num_filters:
            continue

        # Place {'<=', '>=', '='} on numericals and '=' on categoricals.
        idxs = rng.choice(non_null_indices, replace=False, size=num_filters)
        vals = tup[idxs].values
        for i,val in enumerate(vals):
            if isinstance(val, str):
                val = val.replace(',','\,')
                vals[i] = val
        cols = np.take(content_cols, idxs)
        ops = rng.choice(['<=', '>=', '='], size=num_filters)
        sensible_to_do_range = [c in categoricals for c in cols]
        ops = np.where(sensible_to_do_range, ops, '=')

        predicate_list.append( (cols,ops,vals) )
        predicate_strings.append(','.join(
            [','.join((f"{TPCDS_TABLE_TO_ALIAS[c.split('.')[0]]}.{c.split('.')[1]}", o, str(v))) for c, o, v in zip(cols, ops, vals)]))

    print(f"df : {len(sampled_df)}  \t len pred {len(predicate_strings)}")
    return predicate_list,predicate_strings




def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def MakeTablesKey(table_names):
    sorted_tables = sorted(table_names)
    return '-'.join(sorted_tables)


def main(template_csv, output_csv, root, use_cols, num_queries):
    sep ='|'
    tables = datasets.LoadTPCDS(use_cols=use_cols)
    print(f"\nLoad templates : {template_csv}\n")

    specs = get_join_spec_from_file(template_csv, root)
    print("Get join spec")

    rng = np.random.RandomState(1234)

    # Disambiguate to not prune away stuff during join sampling.
    for table_name, table in tables.items():
        for col in table.columns:
            col.name = common.JoinTableAndColumnNames(table.name,
                                                      col.name,
                                                      sep='.')
        table.data.columns = [col.name for col in table.columns]

    print(f"\nSave queries to {output_csv}\n")


    total_time = 0
    for i, join_spec in enumerate(specs):
        assert num_queries > 0

        tables_in_templates = [tables[n] for n in join_spec.join_tables]
        print(f"\n\n===== # of Join : {i + 1}=====\n\n")

        try:
            t1 = time.time()
            MakeQueries(join_spec=join_spec, tables_in_templates=tables_in_templates, use_cols=use_cols,
                                 num_queries=num_queries,
                                 rng=rng, output_file=output_csv,sep=sep)
            excute_time = time.time()-t1
            total_time += excute_time
            print(f"{i+1} done. takes {excute_time:.2f}s \t total :{total_time}")

        except :
            txt =f"----- error num:{i}\n"
            with open("make_error_log.txt",'at') as writer:
                writer.write(txt)

            txt = f"{sep}{sep}{sep}_\n"
            with open(output_csv,'at') as writer:
                writer.write(txt)

        if i%20 == 0:
            shutil.rmtree("./cache/")
            print("remove cache")

    print("Queries are generated")

def get_root_from_file(path):
    root = None
    with open(path,'r') as file:
        txt = file.readline()
        root = txt.split('.')[0]
    assert root is not None
    return root

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--template_csv', nargs="+",help='tmp file name')
    parser.add_argument('--root', default='item')
    parser.add_argument('--use_cols', default='tpcds-db')
    parser.add_argument('--num_queries', help='Number of queries', type=int, default=1)
    args = parser.parse_args()

    template_csv = args.template_csv
    # output_csv = args.output_csv
    # root = args.root
    # use_cols = args.use_cols
    # num_queries = args.num_queries
    # workload = ''
    for input_csv in template_csv:
        t1 = time.time()
        assert '.template' in input_csv
        output_csv = input_csv.replace('.template','.csv')
        root = args.root
        use_cols = 'tpcds-db'
        num_queries = 1

        print(f"use root : {root} \nuse cols : {use_cols}")
        main(input_csv, output_csv, root, use_cols, num_queries)
        print(f"{input_csv} done. takes {time.time()-t1:.2f}s")
    # main(template_csv, output_csv, root, use_cols,num_queries,workload)


