"""Experiment configurations.

EXPERIMENT_CONFIGS holds all registered experiments.

TEST_CONFIGS (defined at end of file) stores "unit tests": these are meant to
run for a short amount of time and to assert metrics are reasonable.

Experiments registered here can be launched using:

  >> python run.py --run <config> [ <more configs> ]
  >> python run.py  # Runs all tests in TEST_CONFIGS.
"""
import os

from ray import tune

EXPERIMENT_CONFIGS = {}
TEST_CONFIGS = {}

# Common config. Each key is auto set as an attribute (i.e. NeuroCard.<attr>)
# so try to avoid any name conflicts with members of that class.
BASE_CONFIG = {
    'cwd': os.getcwd(),
    'epochs_per_iteration': 1,
    'num_eval_queries_per_iteration': 100,
    'num_eval_queries_at_end': 2000,  # End of training.
    'num_eval_queries_at_checkpoint_load': 10000,  # Evaluate a loaded ckpt.
    'epochs': 10,
    'seed': None,
    'order_seed': None,
    'bs': 2048,
    'order': None,
    'layers': 2,
    'fc_hiddens': 128,
    'warmups': 1000,
    'constant_lr': None,
    'lr_scheduler': None,
    'custom_lr_lambda': None,
    'optimizer': 'adam',
    'residual': True,
    'direct_io': True,
    'input_encoding': 'embed',
    'output_encoding': 'embed',
    'query_filters': [5, 12],
    'force_query_cols': None,
    'embs_tied': True,
    'embed_size': 32,
    'input_no_emb_if_leq': True,
    'resmade_drop_prob': 0.,

    # Multi-gpu data parallel training.
    'use_data_parallel': False,

    # If set, load this checkpoint and run eval immediately. No training. Can
    # be glob patterns.
    # Example:
    # 'checkpoint_to_load': tune.grid_search([
    #     'models/*52.006*',
    #     'models/*43.590*',
    #     'models/*42.251*',
    #     'models/*41.049*',
    # ]),
    'checkpoint_to_load': None,
    # Dropout for wildcard skipping.
    'disable_learnable_unk': False,
    'per_row_dropout': True,
    'dropout': 1,
    'table_dropout': False,
    'fixed_dropout_ratio': False,
    'asserts': None,
    'special_orders': 0,
    'special_order_seed': 0,
    'join_tables': [],
    'label_smoothing': 0.0,
    'compute_test_loss': False,

    # Column factorization.
    'factorize': False,
    'factorize_blacklist': None,
    'grouped_dropout': True,
    'factorize_fanouts': False,

    # Eval.
    'eval_psamples': [100, 1000, 10000],
    'eval_join_sampling': None,  # None, or #samples/query.

    # Transformer.
    'use_transformer': False,
    'transformer_args': {},

    # Checkpoint.
    'save_checkpoint_at_end': True,
    'checkpoint_every_epoch': False,

    # Experimental.
    '_save_samples': None,
    '_load_samples': None,
    'num_orderings': 1,
    'num_dmol': 0,

    'mode' : 'TRAIN',
    'save_eval_result' : True,
    'rust_random_seed' : 0, # 0 make non-deterministic
    'data_dir': '../../datasets/imdb/',
    'epoch' :0,
    'sep' : '#',
    'verbose_mode' : False,
    'accum_iter' : 1,
}

JOB_TOY = {
    'dataset': 'imdb',
    'join_tables': [
        'movie_companies','movie_keyword', 'title',
    ],
    'join_keys': {
        'movie_companies': ['movie_id'],
        'movie_keyword': ['movie_id'],
        'title': ['id'],
    },
    'join_clauses': [
        'title.id=movie_companies.movie_id',
        'title.id=movie_keyword.movie_id',
    ],
    'join_how': 'outer',
    'join_name': 'job-toy',
    # See datasets.py.
    'use_cols': 'toy',
    'queries_csv': '../../workloads/NeuroCard/job-light.csv',
    'data_dir': '../../datasets/imdb/',
    'epochs' : 2,
    'bs' : 128,
    'max_steps': 10,
    'num_eval_queries_per_iteration': 2,
    'use_data_parallel' : True,
    'compute_test_loss': True
}
JOB_LIGHT_BASE = {
    'dataset': 'imdb',
    'join_tables': [
        'cast_info', 'movie_companies', 'movie_info', 'movie_keyword', 'title',
        'movie_info_idx'
    ],
    'join_keys': {
        'cast_info': ['movie_id'],
        'movie_companies': ['movie_id'],
        'movie_info': ['movie_id'],
        'movie_keyword': ['movie_id'],
        'title': ['id'],
        'movie_info_idx': ['movie_id']
    },
    # Sampling starts at this table and traverses downwards in the join tree.
    'join_root': 'title',
    # Inferred.
    'join_clauses': None,
    'join_how': 'outer',
    # Used for caching metadata.  Each join graph should have a unique name.
    'join_name': 'job-light',
    # See datasets.py.
    'use_cols': 'simple',
    'seed': 0,
    'per_row_dropout': False,
    'table_dropout': True,
    'embs_tied': True,
    # Num tuples trained =
    #   bs (batch size) * max_steps (# batches per "epoch") * epochs.
    'epochs': 1,
    'bs': 2048,
    'max_steps': 500,
    # Use this fraction of total steps as warmups.
    'warmups': 0.05,
    # Number of DataLoader workers that perform join sampling.
    'loader_workers': 8,
    # Options: factorized_sampler, fair_sampler (deprecated).
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 1024 * 4,
    'layers': 4,
    # Eval:
    'compute_test_loss': True,
    'queries_csv': '../../workloads/NeuroCard/job-light.csv',
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 70,
    'eval_psamples': [4000],

    # Multi-order.
    'special_orders': 0,
    'order_content_only': True,
    'order_indicators_at_front': False,
}

FACTORIZE = {
    'factorize': True,
    'word_size_bits': 10,
    'grouped_dropout': True,
}
TOY_TEST = {
    'join_tables' : ['A','B','C'],
    'join_keys' : { 'A' : ['x'], 'B':['x','y'], 'C':['y']},
    'join_clauses' : [ 'A.x=B.x', 'B.y=C.y'],
    'join_root': 'A',
    'join_how': 'outer',
    'join_name': 'toy-test',
    'use_cols': 'toy_test_col',
    'epochs': 3,
    'bs': 1,
    'resmade_drop_prob': 0.1,
    'max_steps': 50,
    'loader_workers': 8,
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 1,
    'warmups': 0.15,
    # Eval:
    'compute_test_loss': False,
    'queries_csv': '../../workloads/NeuroCard/toy_test.csv',
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 30,
    'eval_psamples': [5],
    'data_dir': '../../datasets/test/',
    'sep': '#',
    'dataset':'toy',
    'use_data_parallel':True,
}


JOB_M = {
    'join_tables': [
        'title', 'aka_title', 'cast_info', 'complete_cast', 'movie_companies',
        'movie_info', 'movie_info_idx', 'movie_keyword', 'movie_link',
        'kind_type', 'comp_cast_type', 'company_name', 'company_type',
        'info_type', 'keyword', 'link_type'
    ],
    'join_keys': {
        'title': ['id', 'kind_id'],
        'aka_title': ['movie_id'],
        'cast_info': ['movie_id'],
        'complete_cast': ['movie_id', 'subject_id'],
        'movie_companies': ['company_id', 'company_type_id', 'movie_id'],
        'movie_info': ['movie_id'],
        'movie_info_idx': ['info_type_id', 'movie_id'],
        'movie_keyword': ['keyword_id', 'movie_id'],
        'movie_link': ['link_type_id', 'movie_id'],
        'kind_type': ['id'],
        'comp_cast_type': ['id'],
        'company_name': ['id'],
        'company_type': ['id'],
        'info_type': ['id'],
        'keyword': ['id'],
        'link_type': ['id']
    },
    'join_clauses': [
        'title.id=aka_title.movie_id',
        'title.id=cast_info.movie_id',
        'title.id=complete_cast.movie_id',
        'title.id=movie_companies.movie_id',
        'title.id=movie_info.movie_id',
        'title.id=movie_info_idx.movie_id',
        'title.id=movie_keyword.movie_id',
        'title.id=movie_link.movie_id',
        'title.kind_id=kind_type.id',
        'comp_cast_type.id=complete_cast.subject_id',
        'company_name.id=movie_companies.company_id',
        'company_type.id=movie_companies.company_type_id',
        'movie_info_idx.info_type_id=info_type.id',
        'keyword.id=movie_keyword.keyword_id',
        'link_type.id=movie_link.link_type_id',
    ],
    'join_root': 'title',
    'join_how': 'outer',
    'join_name': 'job-m',
    'use_cols': 'multi',
    'epochs': 10,
    'bs': 1000,
    'resmade_drop_prob': 0.1,
    'max_steps': 1000,
    'loader_workers': 8,
    'sampler': 'factorized_sampler',
    'sampler_batch_size': 1024 * 16,
    'warmups': 0.15,
    # Eval:



    'compute_test_loss': False,
    'queries_csv': '../../workloads/NeuroCard/job-m.csv',
    'num_eval_queries_per_iteration': 0,
    'num_eval_queries_at_end': 113,
    'eval_psamples': [1000],


}

JOB_M_FACTORIZED = {
    'factorize': True,
    'factorize_blacklist': [],
    'factorize_fanouts': True,
    'word_size_bits': 14,
    'bs': 2048,
    'max_steps': 512,
    'epochs': 20,
    'checkpoint_every_epoch': True,
    'epochs_per_iteration': 1,
}

TPCDS_DEFAULT = dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
                  **JOB_M_FACTORIZED),
    **{
        'dataset': 'tpcds',
        'data_dir': '../../datasets/tpcds/',
        'epoch' : 0,
        'epochs': 1000,
        'bs': 256,
        'max_steps': tune.grid_search([4000]),
        'eval_psamples': [1000],
        'sep' : '|',
        'use_data_parallel':True,
    }
)

TPCDS_TOY = dict(TPCDS_DEFAULT,
    **{
        'dataset': 'tpcds',
        'join_tables': ['item','store_sales',  'store_returns'],
        'join_keys': {'store_sales': ['ss_item_sk'], 'item': ['i_item_sk'], 'store_returns': ['sr_item_sk']},
        'join_root': 'item',
        'join_clauses': ['item.i_item_sk=store_returns.sr_item_sk','item.i_item_sk=store_sales.ss_item_sk'],
        'use_cols': 'tpcds-toy',
        'data_dir': '../../datasets/tpcds/',
        'join_name': 'tpcds-db-tree-i',
        'accum_iter' : 8,



        'queries_csv': '../../workloads/NeuroCard/tpcds-sub.csv',
        # 'factorize_blacklist' : ['__fanout_store_sales', '__fanout_store_sales__ss_item_sk', '__fanout_item', '__fanout_item__i_item_sk', '__fanout_store_returns', '__fanout_store_returns__sr_item_sk', '__fanout_store_returns__sr_returned_date_sk', '__fanout_store_returns__sr_return_time_sk', '__fanout_store_returns__sr_customer_sk', '__fanout_store_returns__sr_cdemo_sk', '__fanout_store_returns__sr_hdemo_sk', '__fanout_store_returns__sr_addr_sk', '__fanout_store_returns__sr_store_sk', '__fanout_store_returns__sr_reason_sk', '__fanout_catalog_sales', '__fanout_catalog_sales__cs_item_sk', '__fanout_catalog_sales__cs_call_center_sk', '__fanout_catalog_sales__cs_catalog_page_sk', '__fanout_catalog_sales__cs_ship_mode_sk', '__fanout_catalog_sales__cs_warehouse_sk', '__fanout_catalog_returns', '__fanout_catalog_returns__cr_item_sk', '__fanout_web_sales', '__fanout_web_sales__ws_item_sk', '__fanout_web_sales__ws_web_page_sk', '__fanout_web_sales__ws_web_site_sk', '__fanout_web_returns', '__fanout_web_returns__wr_item_sk', '__fanout_inventory', '__fanout_inventory__inv_item_sk', '__fanout_promotion', '__fanout_promotion__p_item_sk', '__fanout_date_dim', '__fanout_date_dim__d_date_sk', '__fanout_time_dim', '__fanout_time_dim__t_time_sk', '__fanout_customer', '__fanout_customer__c_customer_sk', '__fanout_customer_demographics', '__fanout_customer_demographics__cd_demo_sk', '__fanout_household_demographics', '__fanout_household_demographics__hd_demo_sk', '__fanout_household_demographics__hd_income_band_sk', '__fanout_customer_address', '__fanout_customer_address__ca_address_sk', '__fanout_store', '__fanout_store__s_store_sk', '__fanout_reason', '__fanout_reason__r_reason_sk', '__fanout_call_center', '__fanout_call_center__cc_call_center_sk', '__fanout_catalog_page', '__fanout_catalog_page__cp_catalog_page_sk', '__fanout_ship_mode', '__fanout_ship_mode__sm_ship_mode_sk', '__fanout_warehouse', '__fanout_warehouse__w_warehouse_sk', '__fanout_web_page', '__fanout_web_page__wp_web_page_sk', '__fanout_web_site', '__fanout_web_site__web_site_sk', '__fanout_income_band', '__fanout_income_band__ib_income_band_sk']
        # 'factorize_blacklist' : ['__fanout_store_sales__ss_item_sk','__fanout_item__i_item_sk','__fanout_store_returns__sr_item_sk', '__fanout_store_returns__sr_returned_date_sk', '__fanout_store_returns__sr_return_time_sk', '__fanout_store_returns__sr_customer_sk', '__fanout_store_returns__sr_cdemo_sk', '__fanout_store_returns__sr_hdemo_sk', '__fanout_store_returns__sr_addr_sk', '__fanout_store_returns__sr_store_sk', '__fanout_store_returns__sr_reason_sk','__fanout_catalog_sales__cs_item_sk', '__fanout_catalog_sales__cs_call_center_sk', '__fanout_catalog_sales__cs_catalog_page_sk', '__fanout_catalog_sales__cs_ship_mode_sk', '__fanout_catalog_sales__cs_warehouse_sk','__fanout_catalog_returns__cr_item_sk','__fanout_web_sales__ws_item_sk', '__fanout_web_sales__ws_web_page_sk', '__fanout_web_sales__ws_web_site_sk','__fanout_web_returns__wr_item_sk','__fanout_inventory__inv_item_sk','__fanout_promotion__p_item_sk','__fanout_date_dim__d_date_sk','__fanout_time_dim__t_time_sk','__fanout_customer__c_customer_sk','__fanout_customer_demographics__cd_demo_sk','__fanout_household_demographics__hd_demo_sk', '__fanout_household_demographics__hd_income_band_sk','__fanout_customer_address__ca_address_sk','__fanout_store__s_store_sk','__fanout_reason__r_reason_sk','__fanout_call_center__cc_call_center_sk','__fanout_catalog_page__cp_catalog_page_sk','__fanout_ship_mode__sm_ship_mode_sk','__fanout_warehouse__w_warehouse_sk','__fanout_web_page__wp_web_page_sk','__fanout_web_site__web_site_sk','__fanout_income_band__ib_income_band_sk']
        # 'factorize_blacklist': ['store_sales:ss_item_sk', 'item:i_item_sk', 'store_returns:sr_item_sk', 'store_returns:sr_returned_date_sk', 'store_returns:sr_return_time_sk', 'store_returns:sr_customer_sk', 'store_returns:sr_cdemo_sk', 'store_returns:sr_hdemo_sk', 'store_returns:sr_addr_sk', 'store_returns:sr_store_sk', 'store_returns:sr_reason_sk', 'catalog_sales:cs_item_sk', 'catalog_sales:cs_call_center_sk', 'catalog_sales:cs_catalog_page_sk', 'catalog_sales:cs_ship_mode_sk', 'catalog_sales:cs_warehouse_sk', 'catalog_returns:cr_item_sk', 'web_sales:ws_item_sk', 'web_sales:ws_web_page_sk', 'web_sales:ws_web_site_sk', 'web_returns:wr_item_sk', 'inventory:inv_item_sk', 'promotion:p_item_sk', 'date_dim:d_date_sk', 'time_dim:t_time_sk', 'customer:c_customer_sk', 'customer_demographics:cd_demo_sk', 'household_demographics:hd_demo_sk', 'household_demographics:hd_income_band_sk', 'customer_address:ca_address_sk', 'store:s_store_sk', 'reason:r_reason_sk', 'call_center:cc_call_center_sk', 'catalog_page:cp_catalog_page_sk', 'ship_mode:sm_ship_mode_sk', 'warehouse:w_warehouse_sk', 'web_page:wp_web_page_sk', 'web_site:web_site_sk', 'income_band:ib_income_band_sk']
        })


JOB_UNION = {
    'join_tables': [
        'title', 'aka_title', 'cast_info', 'complete_cast', 'movie_companies',
        'movie_info', 'movie_info_idx', 'movie_keyword', 'movie_link',
        'kind_type', 'comp_cast_type', 'company_name', 'company_type',
        'info_type', 'keyword', 'link_type'
    ],
    'join_keys': {
        'title': ['id', 'kind_id'],
        'aka_title': ['movie_id'],
        'cast_info': ['movie_id'],
        'complete_cast': ['movie_id', 'subject_id'],
        'movie_companies': ['company_id', 'company_type_id', 'movie_id'],
        'movie_info': ['movie_id'],
        'movie_info_idx': ['info_type_id', 'movie_id'],
        'movie_keyword': ['keyword_id', 'movie_id'],
        'movie_link': ['link_type_id', 'movie_id'],
        'kind_type': ['id'],
        'comp_cast_type': ['id'],
        'company_name': ['id'],
        'company_type': ['id'],
        'info_type': ['id'],
        'keyword': ['id'],
        'link_type': ['id']
    },
    'join_clauses': [
        'title.id=aka_title.movie_id',
        'title.id=cast_info.movie_id',
        'title.id=complete_cast.movie_id',
        'title.id=movie_companies.movie_id',
        'title.id=movie_info.movie_id',
        'title.id=movie_info_idx.movie_id',
        'title.id=movie_keyword.movie_id',
        'title.id=movie_link.movie_id',
        'title.kind_id=kind_type.id',
        'comp_cast_type.id=complete_cast.subject_id',
        'company_name.id=movie_companies.company_id',
        'company_type.id=movie_companies.company_type_id',
        'movie_info_idx.info_type_id=info_type.id',
        'keyword.id=movie_keyword.keyword_id',
        'link_type.id=movie_link.link_type_id',
    ],
    'join_root': 'title',
    'join_how': 'outer',
    'join_name': 'job-union',
    # 'use_cols': 'multi-add-key',
    # 'epochs': 200,
    # 'bs': 2048,
    # 'max_steps': 512,
    # 'loader_workers': 8,
    # 'sampler': 'factorized_sampler',
    # 'sampler_batch_size': 1024 * 16,
    # 'compute_test_loss': False,
    # 'queries_csv': '../../workloads/NeuroCard/job-light-ranges.csv',
    # 'num_eval_queries_per_iteration': 0,
    # 'num_eval_queries_at_end': 1000,
    'eval_psamples': [512],
    'factorize_blacklist' :['__fanout_title', '__fanout_title__id', '__fanout_title__kind_id', '__fanout_aka_title', '__fanout_aka_title__movie_id', '__fanout_cast_info', '__fanout_cast_info__movie_id', '__fanout_complete_cast', '__fanout_complete_cast__movie_id', '__fanout_complete_cast__subject_id', '__fanout_movie_companies', '__fanout_movie_companies__company_id', '__fanout_movie_companies__company_type_id', '__fanout_movie_companies__movie_id', '__fanout_movie_info', '__fanout_movie_info__movie_id', '__fanout_movie_info_idx', '__fanout_movie_info_idx__info_type_id', '__fanout_movie_info_idx__movie_id', '__fanout_movie_keyword', '__fanout_movie_keyword__keyword_id', '__fanout_movie_keyword__movie_id', '__fanout_movie_link', '__fanout_movie_link__link_type_id', '__fanout_movie_link__movie_id', '__fanout_kind_type', '__fanout_kind_type__id', '__fanout_comp_cast_type', '__fanout_comp_cast_type__id', '__fanout_company_name', '__fanout_company_name__id', '__fanout_company_type', '__fanout_company_type__id', '__fanout_info_type', '__fanout_info_type__id', '__fanout_keyword', '__fanout_keyword__id', '__fanout_link_type', '__fanout_link_type__id'],
    'data_dir' : '../../datasets/imdb/',

    'use_cols': 'union',

}

JOB_UNION_TUNE = {
    'max_steps':512,
    'use_data_parallel':False,
    'compute_test_loss': False,
    'num_eval_queries_per_iteration':0,
    'fc_hiddens': 2048,
    'layers' : 4,
    'epochs':50,
    'embed_size' : 32,
    'word_size_bits':14,
    'checkpoint_every_epoch' : False,
    'bs': 256,
}


IMDB_FULL = {
        'join_tables': ['title', 'aka_title', 'movie_link', 'cast_info', 'movie_info', 'movie_info_idx', 'kind_type', 'movie_keyword', 'movie_companies', 'complete_cast', 'link_type', 'char_name', 'role_type', 'name', 'info_type', 'keyword', 'company_name', 'company_type', 'comp_cast_type', 'aka_name', 'person_info'],
        'join_keys':{'title': ['id', 'kind_id'],
                     'aka_title': ['movie_id'],
                     'movie_link': ['link_type_id', 'movie_id'],
                     'cast_info': ['person_id', 'person_role_id', 'movie_id', 'role_id'],
                     'movie_info': ['movie_id'],
                     'movie_info_idx': ['movie_id', 'info_type_id'],
                     'kind_type': ['id'],
                     'movie_keyword': ['keyword_id', 'movie_id'],
                     'movie_companies': ['company_id', 'movie_id', 'company_type_id'],
                     'complete_cast': ['movie_id','subject_id'],
                     'link_type': ['id'],
                     'char_name': ['id'],
                     'role_type': ['id'],
                     'name': ['id'],
                     'info_type': ['id'],
                     'keyword': ['id'],
                     'company_name': ['id'],
                     'company_type': ['id'],
                     'comp_cast_type': ['id'],
                     'aka_name': ['person_id'],
                     'person_info': ['person_id']},
        'join_root': 'title',
        'join_clauses':
                    ['title.id=aka_title.movie_id',
                     'title.id=movie_link.movie_id',
                     'title.id=cast_info.movie_id',
                     'title.id=movie_info.movie_id',
                     'title.id=movie_info_idx.movie_id',
                     'title.kind_id=kind_type.id',
                     'title.id=movie_keyword.movie_id',
                     'title.id=movie_companies.movie_id',
                     'title.id=complete_cast.movie_id',
                     'movie_link.link_type_id=link_type.id',
                     'cast_info.person_role_id=char_name.id',
                     'cast_info.role_id=role_type.id',
                     'cast_info.person_id=name.id',
                     'movie_info_idx.info_type_id=info_type.id',
                     'movie_keyword.keyword_id=keyword.id',
                     'movie_companies.company_id=company_name.id',
                     'movie_companies.company_type_id=company_type.id',
                     'comp_cast_type.id=complete_cast.subject_id',
                     'name.id=aka_name.person_id',
                     'name.id=person_info.person_id'],
        'dataset': 'imdb',
        'use_cols': 'imdb-db',
        'data_dir': '../../datasets/imdb/',
        'join_name': 'imdb-full',
        'queries_csv': '../../workloads/NeuroCard/job-light.csv',
        'accum_iter': 1,
        'layers': 4,
        'epochs':200,
        'loader_workers': 1,
        'compute_test_loss': False,
        'checkpoint_every_epoch': True,
        'eval_psamples': [512],
        'factorize_blacklist':['__fanout_title', '__fanout_title__id', '__fanout_title__kind_id', '__fanout_aka_title', '__fanout_aka_title__movie_id', '__fanout_movie_link', '__fanout_movie_link__link_type_id', '__fanout_movie_link__movie_id', '__fanout_cast_info', '__fanout_cast_info__person_id', '__fanout_cast_info__person_role_id', '__fanout_cast_info__movie_id', '__fanout_cast_info__role_id', '__fanout_movie_info', '__fanout_movie_info__movie_id', '__fanout_movie_info_idx', '__fanout_movie_info_idx__movie_id', '__fanout_movie_info_idx__info_type_id', '__fanout_kind_type', '__fanout_kind_type__id', '__fanout_movie_keyword', '__fanout_movie_keyword__keyword_id', '__fanout_movie_keyword__movie_id', '__fanout_movie_companies', '__fanout_movie_companies__company_id', '__fanout_movie_companies__movie_id', '__fanout_movie_companies__company_type_id', '__fanout_complete_cast', '__fanout_complete_cast__subject_id', '__fanout_complete_cast__movie_id', '__fanout_link_type', '__fanout_link_type__id', '__fanout_char_name', '__fanout_char_name__id', '__fanout_role_type', '__fanout_role_type__id', '__fanout_name', '__fanout_name__id', '__fanout_info_type', '__fanout_info_type__id', '__fanout_keyword', '__fanout_keyword__id', '__fanout_company_name', '__fanout_company_name__id', '__fanout_company_type', '__fanout_company_type__id', '__fanout_comp_cast_type', '__fanout_comp_cast_type__id', '__fanout_aka_name', '__fanout_aka_name__person_id', '__fanout_person_info', '__fanout_person_info__person_id'],

}

IMDB_FULL_TUNE = {

    'use_data_parallel':False,
    'compute_test_loss': False,
    'num_eval_queries_per_iteration':0,

    'checkpoint_every_epoch' : False,
    'embed_size' : 16,
    'fc_hiddens': 2048,
    'word_size_bits': 14,
    'layers' : 4,
    'max_steps':512,
    'accum_iter': 1,
    'epochs' : 100,
    'bs':32,
}



TPCDS_FULL = dict(TPCDS_DEFAULT,
    **{
        'dataset': 'tpcds',
        'join_tables': ['item','store_sales',  'store_returns', 'catalog_sales', 'catalog_returns', 'web_sales', 'web_returns', 'inventory', 'promotion', 'date_dim', 'time_dim', 'customer', 'customer_demographics', 'household_demographics', 'customer_address', 'store', 'reason', 'call_center', 'catalog_page', 'ship_mode', 'warehouse', 'web_page', 'web_site', 'income_band'],
        'join_keys': {'item': ['i_item_sk'], 'catalog_returns': ['cr_item_sk',  'cr_call_center_sk',  'cr_catalog_page_sk',  'cr_refunded_addr_sk',  'cr_returned_date_sk',  'cr_returning_customer_sk'], 'catalog_sales': ['cs_item_sk',  'cs_bill_cdemo_sk',  'cs_bill_hdemo_sk',  'cs_promo_sk',  'cs_ship_mode_sk',  'cs_sold_time_sk',  'cs_warehouse_sk'], 'inventory': ['inv_item_sk'], 'store_returns': ['sr_item_sk', 'sr_reason_sk', 'sr_store_sk'], 'store_sales': ['ss_item_sk'], 'web_returns': ['wr_item_sk', 'wr_web_page_sk'], 'web_sales': ['ws_item_sk', 'ws_web_site_sk'], 'call_center': ['cc_call_center_sk'], 'catalog_page': ['cp_catalog_page_sk'], 'customer_address': ['ca_address_sk'], 'date_dim': ['d_date_sk'], 'customer': ['c_customer_sk'], 'customer_demographics': ['cd_demo_sk'], 'household_demographics': ['hd_demo_sk', 'hd_income_band_sk'], 'promotion': ['p_promo_sk'], 'ship_mode': ['sm_ship_mode_sk'], 'time_dim': ['t_time_sk'], 'warehouse': ['w_warehouse_sk'], 'reason': ['r_reason_sk'], 'store': ['s_store_sk'], 'web_page': ['wp_web_page_sk'], 'web_site': ['web_site_sk'], 'income_band': ['ib_income_band_sk']},
        'join_root': 'item',
        'join_clauses': ['item.i_item_sk=catalog_returns.cr_item_sk','item.i_item_sk=catalog_sales.cs_item_sk','item.i_item_sk=inventory.inv_item_sk','item.i_item_sk=store_returns.sr_item_sk','item.i_item_sk=store_sales.ss_item_sk','item.i_item_sk=web_returns.wr_item_sk','item.i_item_sk=web_sales.ws_item_sk','catalog_returns.cr_call_center_sk=call_center.cc_call_center_sk','catalog_returns.cr_catalog_page_sk=catalog_page.cp_catalog_page_sk','catalog_returns.cr_refunded_addr_sk=customer_address.ca_address_sk','catalog_returns.cr_returned_date_sk=date_dim.d_date_sk','catalog_returns.cr_returning_customer_sk=customer.c_customer_sk','catalog_sales.cs_bill_cdemo_sk=customer_demographics.cd_demo_sk','catalog_sales.cs_bill_hdemo_sk=household_demographics.hd_demo_sk','catalog_sales.cs_promo_sk=promotion.p_promo_sk','catalog_sales.cs_ship_mode_sk=ship_mode.sm_ship_mode_sk','catalog_sales.cs_sold_time_sk=time_dim.t_time_sk','catalog_sales.cs_warehouse_sk=warehouse.w_warehouse_sk','store_returns.sr_reason_sk=reason.r_reason_sk','store_returns.sr_store_sk=store.s_store_sk','web_returns.wr_web_page_sk=web_page.wp_web_page_sk','web_sales.ws_web_site_sk=web_site.web_site_sk','household_demographics.hd_income_band_sk=income_band.ib_income_band_sk'],
        'use_cols': 'tpcds-db',
        'data_dir': '../../datasets/tpcds/',
        'join_name': 'tpcds-full',
        'queries_csv': '../../workloads/NeuroCard/tpcds-sub.csv',
        'factorize_blacklist' : ['__fanout_item', '__fanout_item__i_item_sk', '__fanout_catalog_returns', '__fanout_catalog_returns__cr_item_sk', '__fanout_catalog_returns__cr_call_center_sk', '__fanout_catalog_returns__cr_catalog_page_sk', '__fanout_catalog_returns__cr_refunded_addr_sk', '__fanout_catalog_returns__cr_returned_date_sk', '__fanout_catalog_returns__cr_returning_customer_sk', '__fanout_catalog_sales', '__fanout_catalog_sales__cs_item_sk', '__fanout_catalog_sales__cs_bill_cdemo_sk', '__fanout_catalog_sales__cs_bill_hdemo_sk', '__fanout_catalog_sales__cs_promo_sk', '__fanout_catalog_sales__cs_ship_mode_sk', '__fanout_catalog_sales__cs_sold_time_sk', '__fanout_catalog_sales__cs_warehouse_sk', '__fanout_inventory', '__fanout_inventory__inv_item_sk', '__fanout_store_returns', '__fanout_store_returns__sr_item_sk', '__fanout_store_returns__sr_reason_sk', '__fanout_store_returns__sr_store_sk', '__fanout_store_sales', '__fanout_store_sales__ss_item_sk', '__fanout_web_returns', '__fanout_web_returns__wr_item_sk', '__fanout_web_returns__wr_web_page_sk', '__fanout_web_sales', '__fanout_web_sales__ws_item_sk', '__fanout_web_sales__ws_web_site_sk', '__fanout_call_center', '__fanout_call_center__cc_call_center_sk', '__fanout_catalog_page', '__fanout_catalog_page__cp_catalog_page_sk', '__fanout_customer_address', '__fanout_customer_address__ca_address_sk', '__fanout_date_dim', '__fanout_date_dim__d_date_sk', '__fanout_customer', '__fanout_customer__c_customer_sk', '__fanout_customer_demographics', '__fanout_customer_demographics__cd_demo_sk', '__fanout_household_demographics', '__fanout_household_demographics__hd_demo_sk', '__fanout_household_demographics__hd_income_band_sk', '__fanout_promotion', '__fanout_promotion__p_promo_sk', '__fanout_ship_mode', '__fanout_ship_mode__sm_ship_mode_sk', '__fanout_time_dim', '__fanout_time_dim__t_time_sk', '__fanout_warehouse', '__fanout_warehouse__w_warehouse_sk', '__fanout_reason', '__fanout_reason__r_reason_sk', '__fanout_store', '__fanout_store__s_store_sk', '__fanout_web_page', '__fanout_web_page__wp_web_page_sk', '__fanout_web_site', '__fanout_web_site__web_site_sk', '__fanout_income_band', '__fanout_income_band__ib_income_band_sk'],


 })
TPCDS_FULL_TUNE = {
    'max_steps':512,
    'use_data_parallel':False,
    'compute_test_loss': False,
    'num_eval_queries_per_iteration':0,
    'accum_iter': 1,
    'fc_hiddens': 2048,
    'epochs':150,
    'layers' : 4,
    'embed_size' : 32,
    'word_size_bits':14,
    'checkpoint_every_epoch' : False,
    'bs': 128,
}



TPCDS_BENCHMARK = dict(TPCDS_DEFAULT,
    **{
        'dataset': 'tpcds',
        'join_tables': ['item','store_sales',  'store_returns', 'catalog_sales', 'catalog_returns', 'web_sales', 'web_returns', 'inventory', 'promotion', 'date_dim', 'time_dim', 'customer', 'customer_demographics', 'household_demographics', 'customer_address', 'store', 'reason', 'call_center', 'catalog_page', 'ship_mode', 'warehouse', 'web_page', 'web_site', 'income_band'],
        'join_keys': {'item': ['i_item_sk'], 'catalog_returns': ['cr_item_sk',  'cr_call_center_sk',  'cr_catalog_page_sk',  'cr_refunded_addr_sk',  'cr_returned_date_sk',  'cr_returning_customer_sk'], 'catalog_sales': ['cs_item_sk',  'cs_bill_cdemo_sk',  'cs_bill_hdemo_sk',  'cs_promo_sk',  'cs_ship_mode_sk',  'cs_sold_time_sk',  'cs_warehouse_sk'], 'inventory': ['inv_item_sk'], 'store_returns': ['sr_item_sk', 'sr_reason_sk', 'sr_store_sk'], 'store_sales': ['ss_item_sk'], 'web_returns': ['wr_item_sk', 'wr_web_page_sk'], 'web_sales': ['ws_item_sk', 'ws_web_site_sk'], 'call_center': ['cc_call_center_sk'], 'catalog_page': ['cp_catalog_page_sk'], 'customer_address': ['ca_address_sk'], 'date_dim': ['d_date_sk'], 'customer': ['c_customer_sk'], 'customer_demographics': ['cd_demo_sk'], 'household_demographics': ['hd_demo_sk', 'hd_income_band_sk'], 'promotion': ['p_promo_sk'], 'ship_mode': ['sm_ship_mode_sk'], 'time_dim': ['t_time_sk'], 'warehouse': ['w_warehouse_sk'], 'reason': ['r_reason_sk'], 'store': ['s_store_sk'], 'web_page': ['wp_web_page_sk'], 'web_site': ['web_site_sk'], 'income_band': ['ib_income_band_sk']},
        'join_root': 'item',
        'join_clauses': ['item.i_item_sk=catalog_returns.cr_item_sk','item.i_item_sk=catalog_sales.cs_item_sk','item.i_item_sk=inventory.inv_item_sk','item.i_item_sk=store_returns.sr_item_sk','item.i_item_sk=store_sales.ss_item_sk','item.i_item_sk=web_returns.wr_item_sk','item.i_item_sk=web_sales.ws_item_sk','catalog_returns.cr_call_center_sk=call_center.cc_call_center_sk','catalog_returns.cr_catalog_page_sk=catalog_page.cp_catalog_page_sk','catalog_returns.cr_refunded_addr_sk=customer_address.ca_address_sk','catalog_returns.cr_returned_date_sk=date_dim.d_date_sk','catalog_returns.cr_returning_customer_sk=customer.c_customer_sk','catalog_sales.cs_bill_cdemo_sk=customer_demographics.cd_demo_sk','catalog_sales.cs_bill_hdemo_sk=household_demographics.hd_demo_sk','catalog_sales.cs_promo_sk=promotion.p_promo_sk','catalog_sales.cs_ship_mode_sk=ship_mode.sm_ship_mode_sk','catalog_sales.cs_sold_time_sk=time_dim.t_time_sk','catalog_sales.cs_warehouse_sk=warehouse.w_warehouse_sk','store_returns.sr_reason_sk=reason.r_reason_sk','store_returns.sr_store_sk=store.s_store_sk','web_returns.wr_web_page_sk=web_page.wp_web_page_sk','web_sales.ws_web_site_sk=web_site.web_site_sk','household_demographics.hd_income_band_sk=income_band.ib_income_band_sk'],
        'use_cols': 'tpcds-benchmark',
        'data_dir': '../../datasets/tpcds/',
        'join_name': 'tpcds-benchmark',
        'queries_csv': '../../workloads/NeuroCard/tpcds-sub.csv',
        'factorize_blacklist' : ['__fanout_item', '__fanout_item__i_item_sk', '__fanout_catalog_returns', '__fanout_catalog_returns__cr_item_sk', '__fanout_catalog_returns__cr_call_center_sk', '__fanout_catalog_returns__cr_catalog_page_sk', '__fanout_catalog_returns__cr_refunded_addr_sk', '__fanout_catalog_returns__cr_returned_date_sk', '__fanout_catalog_returns__cr_returning_customer_sk', '__fanout_catalog_sales', '__fanout_catalog_sales__cs_item_sk', '__fanout_catalog_sales__cs_bill_cdemo_sk', '__fanout_catalog_sales__cs_bill_hdemo_sk', '__fanout_catalog_sales__cs_promo_sk', '__fanout_catalog_sales__cs_ship_mode_sk', '__fanout_catalog_sales__cs_sold_time_sk', '__fanout_catalog_sales__cs_warehouse_sk', '__fanout_inventory', '__fanout_inventory__inv_item_sk', '__fanout_store_returns', '__fanout_store_returns__sr_item_sk', '__fanout_store_returns__sr_reason_sk', '__fanout_store_returns__sr_store_sk', '__fanout_store_sales', '__fanout_store_sales__ss_item_sk', '__fanout_web_returns', '__fanout_web_returns__wr_item_sk', '__fanout_web_returns__wr_web_page_sk', '__fanout_web_sales', '__fanout_web_sales__ws_item_sk', '__fanout_web_sales__ws_web_site_sk', '__fanout_call_center', '__fanout_call_center__cc_call_center_sk', '__fanout_catalog_page', '__fanout_catalog_page__cp_catalog_page_sk', '__fanout_customer_address', '__fanout_customer_address__ca_address_sk', '__fanout_date_dim', '__fanout_date_dim__d_date_sk', '__fanout_customer', '__fanout_customer__c_customer_sk', '__fanout_customer_demographics', '__fanout_customer_demographics__cd_demo_sk', '__fanout_household_demographics', '__fanout_household_demographics__hd_demo_sk', '__fanout_household_demographics__hd_income_band_sk', '__fanout_promotion', '__fanout_promotion__p_promo_sk', '__fanout_ship_mode', '__fanout_ship_mode__sm_ship_mode_sk', '__fanout_time_dim', '__fanout_time_dim__t_time_sk', '__fanout_warehouse', '__fanout_warehouse__w_warehouse_sk', '__fanout_reason', '__fanout_reason__r_reason_sk', '__fanout_store', '__fanout_store__s_store_sk', '__fanout_web_page', '__fanout_web_page__wp_web_page_sk', '__fanout_web_site', '__fanout_web_site__web_site_sk', '__fanout_income_band', '__fanout_income_band__ib_income_band_sk'],

    })
TPCDS_BENCHMARK_TUNE = {
    'use_data_parallel':False,
    'compute_test_loss': False,
    'num_eval_queries_per_iteration':0,
    'fc_hiddens': 2048,
    'layers' : 4,
    'epochs':100,
    'embed_size' : 32,
    'word_size_bits':14,
    'checkpoint_every_epoch' : False,
    'bs': 256,
    'queries_csv': '../../workloads/NeuroCard/tpcds-sub.csv',
}



SYN_MULTI = {
        'join_tables': ['table0','table1','table2','table3','table4','table5','table6','table7','table8','table9'],
        'join_keys': {'table0' : ['PK'], 'table1' : ['PK','FK'], 'table2' : ['PK','FK'], 'table3' : ['PK','FK'], 'table4' : ['PK','FK'], 'table5' : ['PK','FK'], 'table6' : ['PK','FK'], 'table7' : ['PK','FK'], 'table8' : ['PK','FK'], 'table9' : ['FK']},
        'join_root': 'table0',
        'join_clauses': ['table0.PK=table1.FK', 'table1.PK=table2.FK', 'table2.PK=table3.FK', 'table3.PK=table4.FK', 'table4.PK=table5.FK', 'table5.PK=table6.FK', 'table6.PK=table7.FK', 'table7.PK=table8.FK', 'table8.PK=table9.FK'],
        'use_cols': 'multi',
        'join_how': 'outer',
        'dataset': 'synthetic',
        'join_name' : 'syn_multi',
        'data_dir': '../../datasets/table10_dom100_skew1.0_seed0/',
        'queries_csv': '../../workloads/NeuroCard/syn_multi_toy.csv' ,
        'eval_psamples': [512],
        'num_eval_queries_at_checkpoint_load': 10000,
        'num_eval_queries_at_end': 10000,
        'factorize_blacklist' : ['__fanout_table0',  '__fanout_table0__PK',  '__fanout_table1',  '__fanout_table1__PK',  '__fanout_table1__FK',  '__fanout_table2',  '__fanout_table2__PK',  '__fanout_table2__FK',  '__fanout_table3',  '__fanout_table3__PK',  '__fanout_table3__FK',  '__fanout_table4',  '__fanout_table4__PK',  '__fanout_table4__FK',  '__fanout_table5',  '__fanout_table5__PK',  '__fanout_table5__FK',  '__fanout_table6',  '__fanout_table6__PK',  '__fanout_table6__FK',  '__fanout_table7',  '__fanout_table7__PK',  '__fanout_table7__FK',  '__fanout_table8',  '__fanout_table8__PK',  '__fanout_table8__FK',  '__fanout_table9',  '__fanout_table9__FK'],
}
SYN_SINGLE = {
        'join_tables': ['table0'],
        'join_keys': {},
        'join_root': 'table0',
        'join_clauses': [],
        'use_cols': 'single',
        'join_how': 'outer',
        'dataset': 'synthetic',
        'join_name' : 'syn_single',
        'data_dir': '../../datasets/col10_dom1000_skew1.0_corr0.8_seed0/',
        'queries_csv': '../../workloads/NeuroCard/syn_single_toy.csv',
        'eval_psamples': [512],
        'sep' :'#',
        'table_dropout' : False,
        'num_eval_queries_at_checkpoint_load': 10000,
}

SYN_SINGLE_TUNE = {
        'epochs': 40,
        'max_steps' : 512,
        'use_data_parallel':False,
        'compute_test_loss': False,
        'num_eval_queries_per_iteration':0,
        'num_eval_queries_at_end':0,
        'fc_hiddens': 2048,
        'layers' : 4,
        'embed_size' : 32,
        'word_size_bits': 14,
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-00.csv',
        'checkpoint_every_epoch' : False,
}

SYN_MULTI_TUNE = {
        'epochs': 40,
        'max_steps' : 512,
        'use_data_parallel':False,
        'compute_test_loss': False,
        'num_eval_queries_per_iteration':0,
        'fc_hiddens': 2048,
        'layers' : 4,
        'embed_size' : 32,
        'word_size_bits': 14,
        'num_eval_queries_at_end':0,
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-00.csv',
        'checkpoint_every_epoch' : False,
}

SYN_MULTI_REVERSE = {
        'join_tables': ['table9', 'table8', 'table7', 'table6', 'table5', 'table4', 'table3', 'table2', 'table1', 'table0'],
        'join_keys': {'table0' : ['PK'], 'table1' : ['PK','FK'], 'table2' : ['PK','FK'], 'table3' : ['PK','FK'], 'table4' : ['PK','FK'], 'table5' : ['PK','FK'], 'table6' : ['PK','FK'], 'table7' : ['PK','FK'], 'table8' : ['PK','FK'], 'table9' : ['FK']},
        'join_root': 'table9',
        'join_clauses': ['table8.PK=table9.FK', 'table7.PK=table8.FK', 'table6.PK=table7.FK', 'table5.PK=table6.FK', 'table4.PK=table5.FK', 'table3.PK=table4.FK', 'table2.PK=table3.FK', 'table1.PK=table2.FK', 'table0.PK=table1.FK'],
        'use_cols': 'multi',
        'join_how': 'outer',
        'dataset': 'synthetic',
        'join_name' : 'syn_multi_rev',
        'data_dir': '../../datasets/table10_dom100_skew1.0_seed0/',
        'queries_csv': '../../workloads/NeuroCard/syn_multi_toy.csv' ,
        'eval_psamples': [512],
        'factorize_blacklist' : ['__fanout_table0',  '__fanout_table0__PK',  '__fanout_table1',  '__fanout_table1__PK',  '__fanout_table1__FK',  '__fanout_table2',  '__fanout_table2__PK',  '__fanout_table2__FK',  '__fanout_table3',  '__fanout_table3__PK',  '__fanout_table3__FK',  '__fanout_table4',  '__fanout_table4__PK',  '__fanout_table4__FK',  '__fanout_table5',  '__fanout_table5__PK',  '__fanout_table5__FK',  '__fanout_table6',  '__fanout_table6__PK',  '__fanout_table6__FK',  '__fanout_table7',  '__fanout_table7__PK',  '__fanout_table7__FK',  '__fanout_table8',  '__fanout_table8__PK',  '__fanout_table8__FK',  '__fanout_table9',  '__fanout_table9__FK'],
}


SYN_MULTI_FANOUT_TEST = {
        'join_tables': ['table2', 'table1', 'table0'],
        'join_keys': {'table0' : ['PK'], 'table1' : ['PK','FK'], 'table2' : ['FK']},
        'join_root': 'table2',
        'join_clauses': ['table1.PK=table2.FK', 'table0.PK=table1.FK'],
        'use_cols': 'multi',
        'join_how': 'outer',
        'dataset': 'synthetic',
        'join_name' : 'syn_multi_fanout',
        'data_dir': '../../datasets/table10_dom100_skew1.0_seed0/',
        'queries_csv': '../../workloads/NeuroCard/fanout_test_10000.csv',
        'eval_psamples': [512],
        'factorize_blacklist' : ['__fanout_table0',  '__fanout_table0__PK',  '__fanout_table1',  '__fanout_table1__PK',  '__fanout_table1__FK',  '__fanout_table2',  '__fanout_table2__PK'],
}

SYN_MULTI_FANOUT_TEST_ORDER = {
        'join_tables': ['table0', 'table1','table2' ],
        'join_keys': {'table0' : ['PK'], 'table1' : ['PK','FK'], 'table2' : ['FK']},
        'join_root': 'table0',
        'join_clauses': ['table0.PK=table1.FK','table1.PK=table2.FK' ],
        'use_cols': 'multi',
        'join_how': 'outer',
        'dataset': 'synthetic',
        'join_name' : 'syn_multi_fanout',
        'data_dir': '../../datasets/table10_dom100_skew1.0_seed0/',
        'queries_csv': '../../workloads/NeuroCard/fanout_test_10000.csv',
        'eval_psamples': [512],
        'factorize_blacklist' : ['__fanout_table0',  '__fanout_table0__PK',  '__fanout_table1',  '__fanout_table1__PK',  '__fanout_table1__FK',  '__fanout_table2',  '__fanout_table2__PK'],
}

IMDB_FULL_DUP = {
        'dataset': 'imdb',
        'join_tables': ['title', 'aka_title', 'movie_link', 'cast_info', 'movie_info', 'movie_info_idx', 'kind_type', 'movie_keyword', 'movie_companies', 'complete_cast', 'link_type', 'char_name', 'role_type', 'name', 'info_type', 'keyword', 'company_name', 'company_type', 'comp_cast_type', 'aka_name', 'person_info', 'movie_link_dup_', 'person_info_dup_', 'info_type_dup_', 'comp_cast_type_dup_'],
        'join_keys': {'title': ['id', 'kind_id'], 'aka_title': ['movie_id'], 'movie_link': ['movie_id', 'link_type_id'], 'cast_info': ['movie_id', 'person_role_id', 'role_id', 'person_id'], 'movie_info': ['movie_id', 'info_type_id'], 'movie_info_idx': ['movie_id', 'info_type_id'], 'kind_type': ['id'], 'movie_keyword': ['movie_id', 'keyword_id'], 'movie_companies': ['movie_id', 'company_id', 'company_type_id'], 'complete_cast': ['movie_id', 'subject_id', 'status_id'], 'link_type': ['id'], 'char_name': ['id'], 'role_type': ['id'], 'name': ['id'], 'info_type': ['id'], 'keyword': ['id'], 'company_name': ['id'], 'company_type': ['id'], 'comp_cast_type': ['id'], 'aka_name': ['person_id'], 'person_info': ['person_id'], 'movie_link_dup_': ['linked_movie_id'], 'person_info_dup_': ['info_type_id'], 'info_type_dup_': ['id'], 'comp_cast_type_dup_': ['id']},
        'join_root': 'title',
        'join_clauses': ['title.id=aka_title.movie_id', 'title.id=movie_link.movie_id', 'title.id=cast_info.movie_id', 'title.id=movie_info.movie_id', 'title.id=movie_info_idx.movie_id', 'title.kind_id=kind_type.id', 'title.id=movie_keyword.movie_id', 'title.id=movie_companies.movie_id', 'title.id=complete_cast.movie_id', 'movie_link.link_type_id=link_type.id', 'cast_info.person_role_id=char_name.id', 'cast_info.role_id=role_type.id', 'cast_info.person_id=name.id', 'movie_info_idx.info_type_id=info_type.id', 'movie_keyword.keyword_id=keyword.id', 'movie_companies.company_id=company_name.id', 'movie_companies.company_type_id=company_type.id', 'complete_cast.subject_id=comp_cast_type.id', 'name.id=aka_name.person_id', 'name.id=person_info.person_id', 'title.id=movie_link_dup_.linked_movie_id', 'info_type.id=person_info_dup_.info_type_id', 'movie_info.info_type_id=info_type_dup_.id', 'complete_cast.status_id=comp_cast_type_dup_.id'],
        'use_cols': 'imdb-full-dup',
        'data_dir': '../../datasets/imdb/',
        'join_name': 'imdb-full-dup',
        'queries_csv': '../../workloads/NeuroCard/job-light.csv',
        'loader_workers': 1,
        'compute_test_loss': False,
        'checkpoint_every_epoch': False,

        'embed_size' : 16,
        'fc_hiddens': 2048,
        'word_size_bits': 14,
        'layers' : 4,
        'max_steps':512,
        'accum_iter': 1,
        'epochs' : 5,
        'bs':32,

        'eval_psamples': [512],
        'factorize_blacklist': ['__fanout_title', '__fanout_title__id', '__fanout_title__kind_id', '__fanout_aka_title', '__fanout_aka_title__movie_id', '__fanout_movie_link', '__fanout_movie_link__movie_id', '__fanout_movie_link__link_type_id', '__fanout_cast_info', '__fanout_cast_info__movie_id', '__fanout_cast_info__person_role_id', '__fanout_cast_info__role_id', '__fanout_cast_info__person_id', '__fanout_movie_info', '__fanout_movie_info__movie_id', '__fanout_movie_info__info_type_id', '__fanout_movie_info_idx', '__fanout_movie_info_idx__movie_id', '__fanout_movie_info_idx__info_type_id', '__fanout_kind_type', '__fanout_kind_type__id', '__fanout_movie_keyword', '__fanout_movie_keyword__movie_id', '__fanout_movie_keyword__keyword_id', '__fanout_movie_companies', '__fanout_movie_companies__movie_id', '__fanout_movie_companies__company_id', '__fanout_movie_companies__company_type_id', '__fanout_complete_cast', '__fanout_complete_cast__movie_id', '__fanout_complete_cast__subject_id', '__fanout_complete_cast__status_id', '__fanout_link_type', '__fanout_link_type__id', '__fanout_char_name', '__fanout_char_name__id', '__fanout_role_type', '__fanout_role_type__id', '__fanout_name', '__fanout_name__id', '__fanout_info_type', '__fanout_info_type__id', '__fanout_keyword', '__fanout_keyword__id', '__fanout_company_name', '__fanout_company_name__id', '__fanout_company_type', '__fanout_company_type__id', '__fanout_comp_cast_type', '__fanout_comp_cast_type__id', '__fanout_aka_name', '__fanout_aka_name__person_id', '__fanout_person_info', '__fanout_person_info__person_id', '__fanout_movie_link_dup_', '__fanout_movie_link_dup___linked_movie_id', '__fanout_person_info_dup_', '__fanout_person_info_dup___info_type_id', '__fanout_info_type_dup_', '__fanout_info_type_dup___id', '__fanout_comp_cast_type_dup_', '__fanout_comp_cast_type_dup___id'],
        }
IMDB_FULL_DUP_TUNE = {
        'embed_size' : 16,
        'fc_hiddens': 2048,
        'word_size_bits': 11,
        'layers' : 4,
        'max_steps':512,
        'accum_iter': 1,
        'epochs' : 100,
        'bs':16,
        'use_data_parallel':False,
        'compute_test_loss': False,
}

### EXPERIMENT CONFIGS ###
EXPERIMENT_CONFIGS = {

    'job-light': dict(
        dict(BASE_CONFIG, **JOB_LIGHT_BASE),
        **{
            'factorize': True,
            'grouped_dropout': True,
            'loader_workers': 4,
            'warmups': 0.05,  # Ignored.
            'lr_scheduler': tune.grid_search(['OneCycleLR-0.28']),
            'loader_workers': 4,
            'max_steps': tune.grid_search([500]),
            'epochs': 7,
            'num_eval_queries_per_iteration': 70,
            'input_no_emb_if_leq': False,
            'eval_psamples': [8000],
            'epochs_per_iteration': 1,
            'resmade_drop_prob': tune.grid_search([.1]),
            'label_smoothing': tune.grid_search([0]),
            'word_size_bits': tune.grid_search([11]),
        }),



    'job-light-rep': dict(
        dict(BASE_CONFIG, **JOB_LIGHT_BASE),
        **{
            'factorize': True,
            'grouped_dropout': True,
            'warmups': 0.05,  # Ignored.
            'lr_scheduler': tune.grid_search(['OneCycleLR-0.28']),
            'loader_workers': 1,
            'max_steps': tune.grid_search([500]),
            'epochs': 7,
            'num_eval_queries_per_iteration': 0,
            'input_no_emb_if_leq': False,
            'eval_psamples': [8000],
            'epochs_per_iteration': 1,
            'resmade_drop_prob': tune.grid_search([.1]),
            'label_smoothing': tune.grid_search([0]),
            'word_size_bits': tune.grid_search([11]),
            'join_pred' : 'False',
        }),




    'job-m-rep': dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
                  **JOB_M_FACTORIZED),**{
        'join_pred' : 'False',
        'num_eval_queries_per_iteration': 0,

    } ),

    # JOB-light-ranges, NeuroCard base.
    'job-light-ranges': dict(
        dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **FACTORIZE),
        **{
            'queries_csv': '../../workloads/NeuroCard/job-light-ranges.csv',
            'use_cols': 'content',
            'num_eval_queries_per_iteration': 1000,
            # 10M tuples total.
            'max_steps': tune.grid_search([500]),
            'epochs': 10,
            # Evaluate after every 1M tuples trained.
            'epochs_per_iteration': 1,
            'loader_workers': 4,
            'eval_psamples': [8000],
            'input_no_emb_if_leq': False,
            'resmade_drop_prob': tune.grid_search([0]),
            'label_smoothing': tune.grid_search([0]),
            'fc_hiddens': 128,
            'embed_size': tune.grid_search([16]),
            'word_size_bits': tune.grid_search([14]),
            'table_dropout': False,
            'lr_scheduler': None,
            'warmups': 0.1,
        },
    ),
    'job-light-ranges-join-pred': dict(
        dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **FACTORIZE),
        **{
            'queries_csv': '../../workloads/NeuroCard/job-light-ranges.csv',
            'use_cols': 'content',
            'num_eval_queries_per_iteration': 1000,
            # 10M tuples total.
            'max_steps': tune.grid_search([500]),
            'epochs': 10,
            # Evaluate after every 1M tuples trained.
            'epochs_per_iteration': 1,
            'loader_workers': 4,
            'eval_psamples': [8000],
            'input_no_emb_if_leq': False,
            'resmade_drop_prob': tune.grid_search([0]),
            'label_smoothing': tune.grid_search([0]),
            'fc_hiddens': 128,
            'embed_size': tune.grid_search([16]),
            'word_size_bits': tune.grid_search([14]),
            'table_dropout': False,
            'lr_scheduler': None,
            'warmups': 0.1,
        },
    ),

    # JOB-M, NeuroCard.
    'job-m': dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
                  **JOB_M_FACTORIZED),
    # JOB-M, NeuroCard-large (Transformer).


    'job-union': dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
                  **JOB_M_FACTORIZED),**JOB_UNION),**{
        'epochs': 50,
        'bs': 256,
        'accum_iter': 1,

        'fc_hiddens': 2048,
        'layers' : 4,
        'embed_size' : 32,
        'word_size_bits':14,
        'use_data_parallel':False,
        'compute_test_loss': False,
        'num_eval_queries_per_iteration':0,
    }),
    'imdb-full' : dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
                  **JOB_M_FACTORIZED),**IMDB_FULL),**{
        # 'embed_size' : 32,
        'embed_size' : 16,
        'fc_hiddens': 2048,
        'word_size_bits': 14,
        'layers' : 3,
        'max_steps':512,
        'accum_iter': 1,
        'epochs' : 100,
        'bs':16,
        'compute_test_loss': False
    }),


    'tpcds-benchmark':  dict(TPCDS_BENCHMARK,**{
        'epochs':100,
        'bs' : 256,
        'layers' : 4,
        'fc_hiddens': 2048,
        'word_size_bits':14,
        'embed_size' : 32,
        'max_steps':512,
        'accum_iter': 1,
        'compute_test_loss': False,

    }),

    'tpcds-full':  dict(TPCDS_BENCHMARK,**{

        'embed_size' : 32,
        'fc_hiddens': 2048,
        'word_size_bits':14,
        'layers' : 4,
        'max_steps':512,
        'accum_iter': 1,
        'epochs':150,
        'bs' : 128,
        'compute_test_loss': False,
        'queries_csv':'../../workloads/NeuroCard/tpcds-sub.csv',
        'use_cols': 'tpcds-db',
        'data_dir': '../../datasets/tpcds/',
        'join_name': 'tpcds-full',
    }),



    ### SYNTHETIC CONFIG

    'syn-single-tuning-test': dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**{
        'epochs': 2,
        'accum_iter': 1,
        'fc_hiddens': 2048,
        'layers' : 4,
        'embed_size' : 32,
        'word_size_bits':14,
        'use_data_parallel':False,
        'compute_test_loss': False,
        'num_eval_queries_per_iteration':0,

    }),
    'syn-single-0625-tuning': dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**{
        'epochs': 200,
        'max_steps' : 512,
        'use_data_parallel':False,
        'compute_test_loss': False,
        'num_eval_queries_per_iteration':0,
        'fc_hiddens': tune.grid_search([2048, 128, 1024,256, 512,64]),
        'layers' : tune.grid_search([4,2,3]),
        'embed_size' : tune.grid_search([16,32]),
        'word_size_bits':tune.grid_search([11, 14 ]),
    }),

    'syn-single': dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**{
        'epochs': 40,
        'max_steps' : 512,
        'use_data_parallel':False,
        'compute_test_loss': False,
        'num_eval_queries_per_iteration':0,
        'fc_hiddens': 2048,
        'layers' : 4,
        'embed_size' : 32,
        'word_size_bits': 14,
    }),




    'syn-multi-tuning-test': dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**{
        'epochs': 2,
        'max_steps' : 10,
        'accum_iter': 1,
        'fc_hiddens': 2048,
        'layers' : 4,
        'embed_size' : 32,
        'word_size_bits':14,
        'use_data_parallel':False,
        'compute_test_loss': False,
        'num_eval_queries_per_iteration':0,

    }),
    'syn-multi-0623-tuning': dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**{
        'epochs': 200,
        'max_steps' : 512,
        'use_data_parallel':False,
        'compute_test_loss': False,
        'num_eval_queries_per_iteration':0,
        'fc_hiddens': tune.grid_search([2048, 128, 1024,256, 512,64]),
        'layers' : tune.grid_search([4,2,3]),
        'embed_size' : tune.grid_search([16,32]),
        'word_size_bits':tune.grid_search([11, 14 ]),

    }),
    'syn-multi': dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**{
        'epochs': 40,
        'max_steps' : 512,
        'use_data_parallel':False,
        'compute_test_loss': False,
        'num_eval_queries_per_iteration':0,
        'fc_hiddens': 2048,
        'layers' : 4,
        'embed_size' : 32,
        'word_size_bits': 14,
    }),

    'syn-single-loss': dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**{
        'epochs': 200,
        'max_steps' : 512,
        'use_data_parallel':False,
        'compute_test_loss': False,
        'num_eval_queries_per_iteration':100,
        'fc_hiddens': 2048,
        'layers' : 4,
        'embed_size' : 32,
        'word_size_bits': 14,
        'queries_csv': '../../workloads/NeuroCard/test/tmp-syn-single.csv',
    }),
    'syn-multi-loss': dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**{
        'epochs': 200,
        'max_steps' : 512,
        'use_data_parallel':False,
        'compute_test_loss': False,
        'num_eval_queries_per_iteration':100,
        'fc_hiddens': 2048,
        'layers' : 4,
        'embed_size' : 32,
        'word_size_bits': 14,
        'queries_csv': '../../workloads/NeuroCard/test/tmp-syn-multi.csv',

    }),

    'syn-single-00': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/00/',
        'join_name': 'syn-single-00',
    }),

    'syn-single-01': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/01/',
        'join_name': 'syn-single-01',
    }),

    'syn-single-02': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/02/',
        'join_name': 'syn-single-02',
    }),

    'syn-single-03': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/03/',
        'join_name': 'syn-single-03',
    }),

    'syn-single-04': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/04/',
        'join_name': 'syn-single-04',
    }),

    'syn-single-05': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/05/',
        'join_name': 'syn-single-05',
    }),

    'syn-single-06': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/06/',
        'join_name': 'syn-single-06',
    }),

    'syn-single-07': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/07/',
        'join_name': 'syn-single-07',
    }),

    'syn-single-08': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/08/',
        'join_name': 'syn-single-08',
    }),

    'syn-single-09': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/09/',
        'join_name': 'syn-single-09',
    }),

    'syn-single-10': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/10/',
        'join_name': 'syn-single-10',
    }),

    'syn-single-11': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/11/',
        'join_name': 'syn-single-11',
    }),

    'syn-single-12': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/12/',
        'join_name': 'syn-single-12',
    }),

    'syn-single-13': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/13/',
        'join_name': 'syn-single-13',
    }),

    'syn-single-14': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/14/',
        'join_name': 'syn-single-14',
    }),

    'syn-single-15': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/15/',
        'join_name': 'syn-single-15',
    }),

    'syn-single-16': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/16/',
        'join_name': 'syn-single-16',
    }),

    'syn-single-17': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/17/',
        'join_name': 'syn-single-17',
    }),

    'syn-single-18': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/18/',
        'join_name': 'syn-single-18',
    }),

    'syn-single-19': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/19/',
        'join_name': 'syn-single-19',
    }),

    'syn-single-20': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/20/',
        'join_name': 'syn-single-20',
    }),

    'syn-single-21': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/21/',
        'join_name': 'syn-single-21',
    }),

    'syn-single-22': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/22/',
        'join_name': 'syn-single-22',
    }),

    'syn-single-23': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_SINGLE),**SYN_SINGLE_TUNE),**{
        'data_dir': '../../datasets/synthetic/single/23/',
        'join_name': 'syn-single-23',
    }),



    'syn-multi-00': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/00/',
        'join_name': 'syn-multi-00',
    }),

    'syn-multi-01': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/01/',
        'join_name': 'syn-multi-01',
    }),

    'syn-multi-02': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/02/',
        'join_name': 'syn-multi-02',
    }),

    'syn-multi-03': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/03/',
        'join_name': 'syn-multi-03',
    }),

    'syn-multi-04': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/04/',
        'join_name': 'syn-multi-04',
    }),

    'syn-multi-05': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/05/',
        'join_name': 'syn-multi-05',
    }),

    'syn-multi-06': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/06/',
        'join_name': 'syn-multi-06',
    }),

    'syn-multi-07': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/07/',
        'join_name': 'syn-multi-07',
    }),

    'syn-multi-08': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/08/',
        'join_name': 'syn-multi-08',
    }),

    'syn-multi-09': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/09/',
        'join_name': 'syn-multi-09',
    }),

    'syn-multi-10': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/10/',
        'join_name': 'syn-multi-10',
    }),

    'syn-multi-11': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/11/',
        'join_name': 'syn-multi-11',
    }),

    'syn-multi-12': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/12/',
        'join_name': 'syn-multi-12',
    }),

    'syn-multi-13': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/13/',
        'join_name': 'syn-multi-13',
    }),

    'syn-multi-14': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/14/','join_name': 'syn-multi-14',
    }),

    'syn-multi-15': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/15/','join_name': 'syn-multi-15',
    }),

    'syn-multi-16': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/16/','join_name': 'syn-multi-16',
    }),

    'syn-multi-17': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/17/','join_name': 'syn-multi-17',
    }),

    'syn-multi-18': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/18/','join_name': 'syn-multi-18',
    }),

    'syn-multi-19': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/19/','join_name': 'syn-multi-19',
    }),

    'syn-multi-20': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/20/','join_name': 'syn-multi-20',
    }),

    'syn-multi-21': dict(dict(dict(dict(dict(dict(BASE_CONFIG, **JOB_LIGHT_BASE), **JOB_M),
              **JOB_M_FACTORIZED), **SYN_MULTI),**SYN_MULTI_TUNE),**{
        'data_dir': '../../datasets/synthetic/multi/21/','join_name': 'syn-multi-21',
    }),
}

#(title, movie_info, movie_info_idx), (title, movie_info), (title, movie_info_idx)

# Run multiple experiments concurrently by using the --run flag, ex:
# $ ./run.py --run job-light

######  TEST CONFIGS ######
# These are run by default if you don't specify --run.

# ------------------------------

TEST_CONFIGS['syn-single-00_syn-single-00'] = dict(
    EXPERIMENT_CONFIGS['syn-single-00'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-00.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-00.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-01_syn-single-01'] = dict(
    EXPERIMENT_CONFIGS['syn-single-01'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-01.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-01.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-02_syn-single-02'] = dict(
    EXPERIMENT_CONFIGS['syn-single-02'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-02.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-02.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-03_syn-single-03'] = dict(
    EXPERIMENT_CONFIGS['syn-single-03'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-03.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-03.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-04_syn-single-04'] = dict(
    EXPERIMENT_CONFIGS['syn-single-04'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-04.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-04.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-05_syn-single-05'] = dict(
    EXPERIMENT_CONFIGS['syn-single-05'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-05.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-05.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-06_syn-single-06'] = dict(
    EXPERIMENT_CONFIGS['syn-single-06'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-06.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-06.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-07_syn-single-07'] = dict(
    EXPERIMENT_CONFIGS['syn-single-07'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-07.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-07.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-08_syn-single-08'] = dict(
    EXPERIMENT_CONFIGS['syn-single-08'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-08.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-08.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-09_syn-single-09'] = dict(
    EXPERIMENT_CONFIGS['syn-single-09'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-09.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-09.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-10_syn-single-10'] = dict(
    EXPERIMENT_CONFIGS['syn-single-10'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-10.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-10.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-11_syn-single-11'] = dict(
    EXPERIMENT_CONFIGS['syn-single-11'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-11.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-11.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-12_syn-single-12'] = dict(
    EXPERIMENT_CONFIGS['syn-single-12'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-12.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-12.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-13_syn-single-13'] = dict(
    EXPERIMENT_CONFIGS['syn-single-13'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-13.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-13.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-14_syn-single-14'] = dict(
    EXPERIMENT_CONFIGS['syn-single-14'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-14.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-14.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-15_syn-single-15'] = dict(
    EXPERIMENT_CONFIGS['syn-single-15'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-15.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-15.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-16_syn-single-16'] = dict(
    EXPERIMENT_CONFIGS['syn-single-16'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-16.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-16.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-17_syn-single-17'] = dict(
    EXPERIMENT_CONFIGS['syn-single-17'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-17.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-17.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-18_syn-single-18'] = dict(
    EXPERIMENT_CONFIGS['syn-single-18'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-18.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-18.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-19_syn-single-19'] = dict(
    EXPERIMENT_CONFIGS['syn-single-19'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-19.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-19.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-20_syn-single-20'] = dict(
    EXPERIMENT_CONFIGS['syn-single-20'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-20.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-20.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-21_syn-single-21'] = dict(
    EXPERIMENT_CONFIGS['syn-single-21'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-21.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-21.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-22_syn-single-22'] = dict(
    EXPERIMENT_CONFIGS['syn-single-22'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-22.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-22.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-single-23_syn-single-23'] = dict(
    EXPERIMENT_CONFIGS['syn-single-23'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-single-test/syn-single-23.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-single-23.tar',
        'eval_psamples': [512],
    })


TEST_CONFIGS['syn-multi-00_syn-multi-00'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-00'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-00.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-00.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-01_syn-multi-01'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-01'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-01.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-01.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-02_syn-multi-02'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-02'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-02.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-02.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-03_syn-multi-03'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-03'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-03.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-03.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-04_syn-multi-04'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-04'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-04.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-04.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-05_syn-multi-05'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-05'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-05.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-05.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-06_syn-multi-06'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-06'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-06.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-06.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-07_syn-multi-07'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-07'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-07.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-07.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-08_syn-multi-08'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-08'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-08.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-08.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-09_syn-multi-09'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-09'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-09.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-09.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-10_syn-multi-10'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-10'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-10.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-10.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-11_syn-multi-11'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-11'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-11.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-11.tar',
        'eval_psamples': [512],
    })
TEST_CONFIGS['syn-multi-12_syn-multi-12'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-12'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-12.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-12.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['syn-multi-13_syn-multi-13'] = dict(
        EXPERIMENT_CONFIGS['syn-multi-13'], **{
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-13.csv',
        'checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-13.tar',
        'eval_psamples': [512],
    })
TEST_CONFIGS['syn-multi-14_syn-multi-14'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-14'], **{
        'num_eval_queries_at_end': 10000,
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-14.csv','checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-14.tar','eval_psamples': [512],    })

TEST_CONFIGS['syn-multi-15_syn-multi-15'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-15'], **{
        'num_eval_queries_at_end': 10000,
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-15.csv','checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-15.tar','eval_psamples': [512],    })

TEST_CONFIGS['syn-multi-16_syn-multi-16'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-16'], **{
        'num_eval_queries_at_end': 21000,
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-16.csv','checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-16.tar','eval_psamples': [512],    })

TEST_CONFIGS['syn-multi-17_syn-multi-17'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-17'], **{
        'num_eval_queries_at_end': 10000,
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-17.csv','checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-17.tar','eval_psamples': [512],    })

TEST_CONFIGS['syn-multi-18_syn-multi-18'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-18'], **{
        'num_eval_queries_at_end': 10000,
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-18.csv','checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-18.tar','eval_psamples': [512],    })

TEST_CONFIGS['syn-multi-19_syn-multi-19'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-19'], **{
        'num_eval_queries_at_end': 10000,
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-19.csv','checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-19.tar','eval_psamples': [512],    })

TEST_CONFIGS['syn-multi-20_syn-multi-20'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-20'], **{
        'num_eval_queries_at_end': 10000,
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-20.csv','checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-20.tar','eval_psamples': [512],    })

TEST_CONFIGS['syn-multi-21_syn-multi-21'] = dict(
    EXPERIMENT_CONFIGS['syn-multi-21'], **{
        'num_eval_queries_at_end': 10000,
        'queries_csv': '../../workloads/NeuroCard/syn-multi-test/syn-multi-21.csv','checkpoint_to_load': '../../models/NeuroCard/synthetic/syn-multi-21.tar','eval_psamples': [512],    })


#############
# exp 1
TEST_CONFIGS['job-light_imdb-small'] = dict(
    EXPERIMENT_CONFIGS['job-light-rep'], **{
        'queries_csv': '../../workloads/NeuroCard/job-light.csv',
        'checkpoint_to_load': '../../models/NeuroCard/job-light.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['job-light_imdb-medium'] = dict(
    EXPERIMENT_CONFIGS['job-union'], **{
        'queries_csv': '../../workloads/NeuroCard/job-light.csv',
        'checkpoint_to_load': '../../models/NeuroCard/job-union.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['job-light_imdb-large'] = dict(
    EXPERIMENT_CONFIGS['imdb-full'], **{
        'queries_csv': '../../workloads/NeuroCard/job-light.csv',
        'checkpoint_to_load': '../../models/NeuroCard/imdb-full.tar',
        'eval_psamples': [512],
    })

# exp 2
TEST_CONFIGS['imdb-small-syn_imdb-small'] = dict(
    EXPERIMENT_CONFIGS['job-light-rep'], **{
        'queries_csv': '../../workloads/NeuroCard/imdb-small-syn.csv',
        'checkpoint_to_load': '../../models/NeuroCard/job-light.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['imdb-small-syn_imdb-medium'] = dict(
    EXPERIMENT_CONFIGS['job-union'], **{
        'queries_csv': '../../workloads/NeuroCard/imdb-small-syn.csv',
        'checkpoint_to_load': '../../models/NeuroCard/job-union.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['imdb-small-syn_imdb-large'] = dict(
    EXPERIMENT_CONFIGS['imdb-full'], **{
        'queries_csv': '../../workloads/NeuroCard/imdb-small-syn.csv',
        'checkpoint_to_load': '../../models/NeuroCard/imdb-full.tar',
        'eval_psamples': [512],
    })


# exp 3,4
TEST_CONFIGS['tpcds-sub_tpcds-bench'] = dict(
    EXPERIMENT_CONFIGS['tpcds-benchmark'], **{
        'queries_csv': '../../workloads/NeuroCard/tpcds-sub.csv',
        'checkpoint_to_load': '../../models/NeuroCard/tpcds-benchmark.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['tpcds-sub_tpcds-large'] = dict(
    EXPERIMENT_CONFIGS['tpcds-benchmark'], **{
        'queries_csv': '../../workloads/NeuroCard/tpcds-sub.csv',
        'checkpoint_to_load': '../../models/NeuroCard/tpcds-benchmark.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['tpcds-large_syn_tpcds-large'] = dict(
    EXPERIMENT_CONFIGS['tpcds-full'], **{
        'queries_csv': '../../workloads/NeuroCard/tpcds-large.csv',
        'checkpoint_to_load': '../../models/NeuroCard/tpcds-full.tar',
        'eval_psamples': [512],
    })


# exp 5
TEST_CONFIGS['imdb-medium-syn_imdb-medium'] = dict(
    EXPERIMENT_CONFIGS['job-union'], **{
        'queries_csv': '../../workloads/NeuroCard/imdb-medium-syn.csv',
        'checkpoint_to_load': '../../models/NeuroCard/job-union.tar',
        'eval_psamples': [512],
    })

TEST_CONFIGS['imdb-medium-syn_imdb-large'] = dict(
    EXPERIMENT_CONFIGS['imdb-full'], **{
        'queries_csv': '../../workloads/NeuroCard/imdb-medium-syn.csv',
        'checkpoint_to_load': '../../models/NeuroCard/imdb-full.tar',
        'eval_psamples': [512],
    })





for name in TEST_CONFIGS:
    TEST_CONFIGS[name].update({'save_checkpoint_at_end': False})
    TEST_CONFIGS[name].update({'mode': 'INFERENCE'})
EXPERIMENT_CONFIGS.update(TEST_CONFIGS)