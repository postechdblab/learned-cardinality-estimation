#!/bin/bash
cd ./MSCN/


#job-light workload
workload=../workloads/NeuroCard/job-light.csv

#generate bitmap feature file
python generate_bitmap.py --input ${workload} --sample ../samples/imdb --dbname imdb

dbname=imdb-small
python run_fcn.py --load ../models/FCN/${dbname}/model --dbname ${dbname} --test ${workload} --output ../results/FCN/job-light_${dbname}.csv

dbname=imdb-medium
python run_fcn_var.py --load ../models/FCN/${dbname}/model --dbname ${dbname} --string --test ${workload} --output ../results/FCN/job-light_${dbname}.csv

dbname=imdb-large
python run_fcn_var.py --load ../models/FCN/${dbname}/model --dbname ${dbname} --string --test ${workload} --output ../results/FCN/job-light_${dbname}.csv



#imdb-small-syn workload
workload=../workloads/NeuroCard/imdb-small-syn.csv

#generate bitmap feature file
python generate_bitmap.py --input ${workload} --sample ../samples/imdb --dbname imdb

dbname=imdb-small
python run_fcn.py --load ../models/FCN/${dbname}/model --dbname ${dbname} --test ${workload} --output ../results/FCN/imdb-small-syn_${dbname}.csv

dbname=imdb-medium
python run_fcn_var.py --load ../models/FCN/${dbname}/model --dbname ${dbname} --string --test ${workload} --output ../results/FCN/imdb-small-syn_${dbname}.csv

dbname=imdb-large
python run_fcn_var.py --load ../models/FCN/${dbname}/model --dbname ${dbname} --string --test ${workload} --output ../results/FCN/imdb-small-syn_${dbname}.csv



#tpcds-large workload
workload=../workloads/NeuroCard/tpcds-large.csv

#generate bitmap feature file
python generate_bitmap.py --input ${workload} --sample ../samples/tpcds --dbname tpcds

dbname=tpcds-large
python run_fcn_var.py --load ../models/FCN/${dbname}/model --dbname ${dbname} --string --test ${workload} --output ../results/FCN/tpcds-large_${dbname}.csv



#imdb-medium-syn workload
workload=../workloads/NeuroCard/imdb-medium-syn.csv

#generate bitmap feature file
python generate_bitmap.py --input ${workload} --sample ../samples/imdb --dbname imdb

dbname=imdb-medium
python run_fcn_var.py --load ../models/FCN/${dbname}/model --dbname ${dbname} --string --test ${workload} --output ../results/FCN/imdb-medium-syn_${dbname}.csv

dbname=imdb-large
python run_fcn_var.py --load ../models/FCN/${dbname}/model --dbname ${dbname} --string --test ${workload} --output ../results/FCN/imdb-medium-syn_${dbname}.csv



#syn-single databases
for i in {00..23}
do
  
  workload=../workloads/NeuroCard/syn-single-test/syn-single-${i}.csv
  
  #generate bitmap feature file
  python generate_bitmap.py --input ${workload} --sample ../samples/synthetic/single/${i} --dbname syn-single
  
  dbname=syn-single-${i}
  python run_fcn.py --load ../models/FCN/syn-single/${dbname}/model --dbname ${dbname} --test ${workload} --output ../results/FCN/syn-single-${i}_${dbname}.csv
  
done



#syn-multi databases
for i in {00..21}
do
  if [ $i -eq 01 ] 
  then
    continue
  fi
  
  workload=../workloads/NeuroCard/syn-multi-test/syn-multi-${i}.csv
  
  #generate bitmap feature file
  python generate_bitmap.py --input ${workload} --sample ../samples/synthetic/multi/${i} --dbname syn-multi
  
  dbname=syn-multi-${i}
  python run_fcn.py --load ../models/FCN/syn-multi/${dbname}/model --dbname ${dbname} --test ${workload} --output ../results/FCN/syn-multi-${i}_${dbname}.csv
  
done
