import argparse
import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import time

import numpy as np
import pandas as pd

from schemas.imdb.schema import gen_job_union_schema, gen_job_original_schema
from schemas.tpc_ds.schema import gen_tpcds_benchmark_schema


from ensemble_compilation.graph_representation import QueryType
from ensemble_compilation.spn_ensemble import read_spn_ensemble
from evaluation.utils import parse_query,new_parse_query
import traceback
import pickle


np.random.seed(1)

parser = argparse.ArgumentParser()

parser.add_argument('--input',
                    type=str,
                    required=True)
parser.add_argument('--output',
                    type=str,
                    required=True)
parser.add_argument('--model',
                    type=str,
                    required=True)
parser.add_argument('--schema',
                    type=str,
                    required=True)
parser.add_argument('--nc',action='store_true')

args = parser.parse_args()

FAIL = -1



class GenCodeStats:

    def __init__(self):
        self.calls = 0
        self.total_time = 0.0


# TODO: get a parameter for which ensemble to use
def evaluate_cardinalities(ensemble_location, physical_db_name, query_filename, target_csv_path, schema,
                           rdc_spn_selection, pairwise_rdc_path, use_generated_code=False,
                           max_variants=1, merge_indicator_exp=False, exploit_overlapping=False, min_sample_ratio=0,meta_path=None):

    # load ensemble
    # TODO: load NC and BN

    t1 = time.time()
    spn_ensemble = read_spn_ensemble(ensemble_location, build_reverse_dict=True)
    # read all queries
    with open(query_filename) as f:
        queries = f.readlines()

    if use_generated_code:
        spn_ensemble.use_generated_code()
    # latencies = []
    est_card_list = list()
    with open(meta_path,'rb') as file:
        meta_data = pickle.load(file)
    
    t2 = time.time()
    print(f"Load {t2-t1:0.2f}s")
    

    t1 = time.time()
    for query_no, query_str in enumerate(queries):

        query_str = query_str.strip()
        query = new_parse_query(query_str.strip(), schema,meta_data)
        if query is None:
            est_card_list.append(FAIL)
            continue

        assert query.query_type == QueryType.CARDINALITY
        
        try:
            # only relevant for generated code
            gen_code_stats = GenCodeStats()

            # card_start_t = perf_counter()
            _, factors, cardinality_predict, factor_values = spn_ensemble \
                .cardinality(query, rdc_spn_selection=rdc_spn_selection, pairwise_rdc_path=pairwise_rdc_path,
                             merge_indicator_exp=merge_indicator_exp, max_variants=max_variants,
                             exploit_overlapping=exploit_overlapping, return_factor_values=True,
                             gen_code_stats=gen_code_stats)
            # card_end_t = perf_counter()
            # latency_ms = (card_end_t - card_start_t) * 1000

            if cardinality_predict is None:
                est_card_list.append(FAIL)
                continue
            cardinality_predict = max(cardinality_predict, 1.0)

            est_card_list.append(cardinality_predict)

        except Exception as error:
            est_card_list.append(FAIL)

    assert len(est_card_list) == len(queries)

    t2 = time.time()
    print(f"infer {t2-t1:0.2f}s")

    with open(target_csv_path,'w') as writer:
        for est_card in est_card_list:
            writer.write(f"{est_card}\n")


if __name__ == '__main__':
    input_path = args.input
    output_path = args.output
    schema_name = args.schema
    model_path = args.model

    if schema_name == 'job-union' :
        csv_path = '../nc_estimator/neurocard/datasets/job_csv_export'
        hdf_path = './hdfs/union_gen_syn_hdf'
        meta_path = hdf_path +'/meta_data.pkl'

        table_csv_path = csv_path + '/{}.csv'
        schema = gen_job_union_schema(table_csv_path)

    elif schema_name == 'job-original':
        csv_path = '../nc_estimator/neurocard/datasets/job_csv_export'
        hdf_path = './hdfs/job_original_gen_syn_hdf'
        meta_path = os.path.join(hdf_path,'/meta_data.pkl')

        table_csv_path = csv_path + '/{}.csv'
        schema = gen_job_original_schema(table_csv_path)


    elif schema_name == 'tpcds-benchmark':
        csv_path = '../nc_estimator/neurocard/datasets/tpcds_2_13_0'
        hdf_path = './hdfs/ben_gen_syn_hdf'
        meta_path = os.path.join(hdf_path,'/meta_data.pkl')
        
        table_csv_path = csv_path + '/{}.csv'
        schema = gen_tpcds_benchmark_schema(table_csv_path)

    else:
        assert False, 'Not supported schema'

    pairwise_rdc_path = os.path.join(model_path,'pairwise_rdc.pkl')

    if args.nc:
        model_path = os.path.join(model_path, 'nc_ensemble/ensemble_join_3_budget_5_10000000.pkl')
    else :
        model_path = os.path.join(model_path, 'spn_ensemble/ensemble_join_3_budget_5_10000000.pkl')

    

    evaluate_cardinalities(model_path, None, input_path, output_path,
                           schema, True, pairwise_rdc_path,
                           use_generated_code=None,
                           merge_indicator_exp=True,
                           exploit_overlapping=True, max_variants=1, min_sample_ratio=0,
                          meta_path =meta_path)