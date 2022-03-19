import numpy as np
import pandas as pd
import FullOuterSampler.common as common
import FullOuterSampler.datasets as datasets
import FullOuterSampler.join_utils as join_utils
import FullOuterSampler.experiments as experiments
from FullOuterSampler.factorized_sampler import FactorizedSamplerIterDataset
import datetime


def get_full_outer_sampler(df_dict,join_clause_list,sample_bs,seed=1234):
	join_tables = list(df_dict.keys())
	join_keys = bulid_join_keys(join_clause_list)
	loaded_tables = list()
	print(join_tables)

	for t in join_tables:
		df = df_dict[t]
		table = datasets.LoadDataset(t,df)
		loaded_tables.append(table)

	join_root = join_tables[0]
    
	now = datetime.datetime.now().strftime('%m%d%H%M')

	join_spec = join_utils.get_join_spec({
		"join_tables": join_tables,
		"join_keys": join_keys,
		"join_root": join_root,
		"join_clauses": join_clause_list,
		"join_how": "outer",
		"join_name": f"fct-{now}"
		})
	print(f"""
    join spec
    	"join_tables": {join_tables},
		"join_keys": {join_keys},
		"join_root": {join_root},
		"join_clauses": {join_clause_list},
		"join_how": "outer",
		"join_name": f"fct-{now}"
    """)
    
	rng = np.random.RandomState(seed)

	ds = FactorizedSamplerIterDataset(loaded_tables,
									  join_spec,
									  df_dict,
									  sample_batch_size=sample_bs,
									  disambiguate_column_names=False,
									  add_full_join_indicators=False,
									  add_full_join_fanouts=False,
									  rust_random_seed=seed,
									  rng=rng)
	return ds.sampler, loaded_tables




def bulid_join_keys(join_clauses):
	join_key_dict = dict()
	result_dict = dict()

	for clause in join_clauses:
		A,B = clause.split('=')
		t1,c1 = A.split('.')
		t2,c2 = B.split('.')
		if t1 not in join_key_dict.keys():
			join_key_dict[t1] = set()
		if t2 not in join_key_dict.keys():
			join_key_dict[t2] = set()
		join_key_dict[t1].add(c1)
		join_key_dict[t2].add(c2)

	for k in join_key_dict.keys():
		result_dict[k] = list(join_key_dict[k])
	return result_dict





# if __name__ == '__main__':

# 	schema = 'job-m' # defined in experiments.py EXPERIMENT_CONFIGS 
# 	sample_bs = 100
# 	seed = 1234
# 	df_dict = dict()
# 	schema_test = experiments.EXPERIMENT_CONFIGS[schema]
# 	join_clause = experiments.EXPERIMENT_CONFIGS[schema]['join_clauses']
	
# 	for table in schema_test['join_tables']:
# 		path = schema_test['data_dir']+table+'.csv'
# 		df_dict[table] = pd.read_csv(path)


	

# 	sampler,_ = get_full_outer_sampler(df_dict,join_clause,sample_bs,dataset='imdb',seed=seed)
# 	print(sampler.run())



