#!/bin/bash
cd ./DeepDB
query_path=../workloads/DeepDB/job-light/sql/job_light_queries.sql
true_card_path=../workloads/DeepDB/job-light/sql/job_light_true_cardinalities.csv

# imdb-small database
dataset=imdb-light

meta_path=../models/DeepDB/imdb-small/imdb-small-DD-JCT/meta_data
model_dir=../models/DeepDB/imdb-small/imdb-small-DD-JCT
rdc_path=$model_dir/pairwise_rdc.pkl
target=../results/DeepDB-JCT/job-light_imdb-small.csv

python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --hdf_path $meta_path \
    --pairwise_rdc_path $rdc_path \
    --dataset $dataset \
    --target_path $target \
    --ensemble_location $model_dir/spn_ensemble/ensemble_join_3_budget_5_10000000.pkl \
    --query_file_location $query_path \
    --ground_truth_file_location $true_card_path 

# imdb-medium database
dataset=job-union

meta_path=../models/DeepDB/imdb-medium/imdb-medium-DD-JCT/meta_data
model_dir=../models/DeepDB/imdb-medium/imdb-medium-DD-JCT
rdc_path=$model_dir/pairwise_rdc.pkl
target=../results/DeepDB-JCT/job-light_imdb-medium.csv

python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --hdf_path $meta_path \
    --pairwise_rdc_path $rdc_path \
    --dataset $dataset \
    --target_path $target \
    --ensemble_location $model_dir/spn_ensemble/ensemble_join_3_budget_5_10000000.pkl \
    --query_file_location $query_path \
    --ground_truth_file_location $true_card_path 


# imdb-large database
dataset=imdb-full

meta_path=../models/DeepDB/imdb-large/imdb-large-DD-JCT/meta_data
model_dir=../models/DeepDB/imdb-large/imdb-large-DD-JCT
rdc_path=$model_dir/pairwise_rdc.pkl
target=../results/DeepDB-JCT/job-light_imdb-large.csv

python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --hdf_path $meta_path \
    --pairwise_rdc_path $rdc_path \
    --dataset $dataset \
    --target_path $target \
    --ensemble_location $model_dir/spn_ensemble/ensemble_join_3_budget_5_10000000.pkl \
    --query_file_location $query_path \
    --ground_truth_file_location $true_card_path 

query_path=../workloads/DeepDB/job-light/sql/job-light-test-1000.sql
true_card_path=../workloads/DeepDB/job-light/sql/job-light-test-1000_true-card.csv

# imdb-small database
dataset=imdb-light

meta_path=../models/DeepDB/imdb-small/imdb-small-DD-JCT/meta_data
model_dir=../models/DeepDB/imdb-small/imdb-small-DD-JCT
rdc_path=$model_dir/pairwise_rdc.pkl
target=../results/DeepDB-JCT/imdb-small-syn_imdb-small.csv

python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --hdf_path $meta_path \
    --pairwise_rdc_path $rdc_path \
    --dataset $dataset \
    --target_path $target \
    --ensemble_location $model_dir/spn_ensemble/ensemble_join_3_budget_5_10000000.pkl \
    --query_file_location $query_path \
    --ground_truth_file_location $true_card_path

# imdb-medium database
dataset=job-union

meta_path=../models/DeepDB/imdb-medium/imdb-medium-DD-JCT/meta_data
model_dir=../models/DeepDB/imdb-medium/imdb-medium-DD-JCT
rdc_path=$model_dir/pairwise_rdc.pkl
target=../results/DeepDB-JCT/imdb-small-syn_imdb-medium.csv

python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --hdf_path $meta_path \
    --pairwise_rdc_path $rdc_path \
    --dataset $dataset \
    --target_path $target \
    --ensemble_location $model_dir/spn_ensemble/ensemble_join_3_budget_5_10000000.pkl \
    --query_file_location $query_path \
    --ground_truth_file_location $true_card_path


# imdb-large database
dataset=imdb-full

meta_path=../models/DeepDB/imdb-large/imdb-large-DD-JCT/meta_data
model_dir=../models/DeepDB/imdb-large/imdb-large-DD-JCT
rdc_path=$model_dir/pairwise_rdc.pkl
target=../results/DeepDB-JCT/imdb-small-syn_imdb-large.csv

python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --hdf_path $meta_path \
    --pairwise_rdc_path $rdc_path \
    --dataset $dataset \
    --target_path $target \
    --ensemble_location $model_dir/spn_ensemble/ensemble_join_3_budget_5_10000000.pkl \
    --query_file_location $query_path \
    --ground_truth_file_location $true_card_path

for M_NUM in "00"  "01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23"; do

dataset=syn-single
meta_path=../models/DeepDB/syn/syn_single_${M_NUM}/meta_data
model_dir=../models/DeepDB/syn/syn_single_${M_NUM}
database_name=single${M_NUM}

query_path=../workloads/DeepDB/syn/sql/syn-single-${M_NUM}.sql
true_card_path=../workloads/DeepDB/syn/sql/syn-single-${M_NUM}_true-card.csv
target=../results/DeepDB-JCT/syn-signle-${M_NUM}_syn-single-${M_NUM}.csv

python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --hdf_path $meta_path \
    --dataset $dataset \
    --target_path $target \
    --ensemble_location $model_dir/ensemble_single_syn-single_1000000.pkl \
    --query_file_location $query_path \
    --ground_truth_file_location $true_card_path

echo "Evaluate cardinality ${M_NUM} from $query_path";

done


for M_NUM in "00" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21"; do
dataset=syn-multi
csv_path=../datasets/synthetic/multi/${M_NUM}


query_path=../workloads/DeepDB/syn/sql/syn-multi-${M_NUM}.sql
true_card_path=../workloads/DeepDB/syn/sql/syn-multi-${M_NUM}_true-card.csv

meta_path=../models/DeepDB/syn/syn_multi_${M_NUM}-jct/meta_data
model_dir=../models/DeepDB/syn/syn_multi_${M_NUM}-jct
rdc_path=$model_dir/pairwise_rdc.pkl
target=../results/DeepDB-JCT/syn-multi-${M_NUM}_syn-multi-${M_NUM}.csv

python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --hdf_path $meta_path \
    --pairwise_rdc_path $rdc_path \
    --dataset $dataset \
    --target_path $target \
    --ensemble_location $model_dir/spn_ensemble/ensemble_join_3_budget_5_10000000.pkl \
    --query_file_location $query_path \
    --ground_truth_file_location $true_card_path

done


query_path=../workloads/DeepDB/job-light/sql/job-light-test-1000.sql
true_card_path=../workloads/DeepDB/job-light/sql/job-light-test-1000_true-card.csv

# imdb-small database
dataset=imdb-light

meta_path=../models/DeepDB/imdb-small/imdb-small-DD-JCT/meta_data
model_dir=../models/DeepDB/imdb-small/SPN-imdb-small-DD-JCT
rdc_path=$model_dir/pairwise_rdc.pkl
target=../results/DeepDB-JCT/imdb-small-syn_SPN-imdb-small.csv

python3 maqp.py --evaluate_cardinalities \
    --rdc_spn_selection \
    --max_variants 1 \
    --hdf_path $meta_path \
    --pairwise_rdc_path $rdc_path \
    --dataset $dataset \
    --target_path $target \
    --ensemble_location $model_dir/spn_ensemble/ensemble_join_3_budget_5_10000000.pkl \
    --query_file_location $query_path \
    --ground_truth_file_location $true_card_path


