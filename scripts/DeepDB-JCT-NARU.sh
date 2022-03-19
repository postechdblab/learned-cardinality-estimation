#!/bin/bash
cd ./DeepDB
query_path=../workloads/DeepDB/job-light/sql/job_light_queries.sql
true_card_path=../workloads/DeepDB/job-light/sql/job_light_true_cardinalities.csv

# imdb-small database
dataset=imdb-light

meta_path=../models/DeepDB/imdb-small/imdb-small-DD-JCT-naru/meta_data
model_dir=../models/DeepDB/imdb-small/imdb-small-DD-JCT-naru
rdc_path=$model_dir/pairwise_rdc.pkl
target=../results/DeepDB-JCT-NARU/job-light_imdb-small.csv

python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --hdf_path $meta_path \
    --pairwise_rdc_path $rdc_path \
    --dataset $dataset \
    --target_path $target \
    --ensemble_location $model_dir/nc_ensemble/ensemble_join_3_budget_5_10000000.pkl \
    --query_file_location $query_path \
    --ground_truth_file_location $true_card_path 

query_path=../workloads/DeepDB/job-light/sql/job-light-test-1000.sql
true_card_path=../workloads/DeepDB/job-light/sql/job-light-test-1000_true-card.csv

target=../results/DeepDB-JCT-NARU/imdb-small-syn_imdb-small.csv

python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --hdf_path $meta_path \
    --pairwise_rdc_path $rdc_path \
    --dataset $dataset \
    --target_path $target \
    --ensemble_location $model_dir/nc_ensemble/ensemble_join_3_budget_5_10000000.pkl \
    --query_file_location $query_path \
    --ground_truth_file_location $true_card_path
