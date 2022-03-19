#!/bin/bash
cd ./E2E/


#job-light workload
workload=../workloads/E2E/job-light/job-light_seq_sample.json

dbname=imdb-small
python test_e2e.py --model ../models/E2E/${dbname}/model  --input ${workload} --output ../results/E2E/job-light_${dbname}.csv

dbname=imdb-medium
python test_e2e.py --model ../models/E2E/${dbname}/model  --input ${workload} --output ../results/E2E/job-light_${dbname}.csv

dbname=imdb-large
python test_e2e.py --model ../models/E2E/${dbname}/model  --input ${workload} --output ../results/E2E/job-light_${dbname}.csv



#imdb-small-syn workload
workload=../workloads/E2E/imdb-small-syn/imdb-small-syn_seq_sample.json

dbname=imdb-small
python test_e2e.py --model ../models/E2E/${dbname}/model --input ${workload} --output ../results/E2E/imdb-small-syn_${dbname}.csv

dbname=imdb-medium
python test_e2e.py --model ../models/E2E/${dbname}/model --input ${workload} --output ../results/E2E/imdb-small-syn_${dbname}.csv

dbname=imdb-large
python test_e2e.py --model ../models/E2E/${dbname}/model --input ${workload} --output ../results/E2E/imdb-small-syn_${dbname}.csv



#imdb-medium-syn workload
workload=../workloads/E2E/imdb-medium-syn/imdb-medium-syn_seq_sample.json

dbname=imdb-medium
python test_e2e.py --model ../models/E2E/${dbname}/model  --input ${workload} --output ../results/E2E/imdb-medium-syn_${dbname}.csv

dbname=imdb-large
python test_e2e.py --model ../models/E2E/${dbname}/model  --input ${workload} --output ../results/E2E/imdb-medium-syn_${dbname}.csv



#syn-single databases
for i in {00..23}
do
  
  workload=../workloads/E2E/syn-single-test/syn-single-${i}_seq_sample.json
  
  dbname=syn-single-${i}
  python test_e2e.py --model ../models/E2E/syn-single/${dbname}/model --input ${workload} --output ../results/E2E/syn-single-${i}_${dbname}.csv
  
done



#syn-multi databases
for i in {00..21}
do
	if [ $i -eq 01 ] 
  then
    continue
  fi

  workload=../workloads/E2E/syn-multi-test/syn-multi-${i}_seq_sample.json
  
  dbname=syn-multi-${i}
  python test_e2e.py --model ../models/E2E/syn-multi/${dbname}/model --input ${workload} --output ../results/E2E/syn-multi-${i}_${dbname}.csv
  
done
