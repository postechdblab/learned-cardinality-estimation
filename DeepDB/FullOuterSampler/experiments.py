
EXPERIMENT_CONFIGS = {
    'job-m' :
    {'join_tables': [
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
         'dataset' : 'imdb',
        'join_root': 'title',
        'join_how': 'outer',
        'use_cols': 'multi',
        'data_dir': 'datasets/job_csv_export/'
        },


    'job-light' :{
        'join_tables': ['title','cast_info', 'movie_companies', 'movie_info', 'movie_keyword', 'movie_info_idx'],
        'join_keys': 
            {'title': ['id'], 'cast_info': ['movie_id'], 'movie_companies': ['movie_id'], 'movie_info': ['movie_id'], 'movie_keyword': ['movie_id'], 'movie_info_idx': ['movie_id']},
            'join_clauses': [
                'title.id=cast_info.movie_id',
                'title.id=movie_companies.movie_id',
                'title.id=movie_info.movie_id',
                'title.id=movie_keyword.movie_id',
                'title.id=movie_info_idx.movie_id'

            ],
        'dataset' : 'imdb',
        'join_root': 'title',
        'join_how': 'outer',
        'use_cols': 'simple',
        'data_dir': 'datasets/job_csv_export/',
        },


    'imdb-full' :{
        'join_tables':['title', 'aka_title', 'cast_info', 'complete_cast', 'movie_companies', 'movie_info', 'movie_info_idx', 'movie_keyword', 'movie_link', 'kind_type', 'char_name', 'role_type', 'aka_name', 'name', 'comp_cast_type', 'company_name', 'company_type', 'info_type', 'keyword', 'link_type', 'person_info'],
        'join_keys': {'name': ['id'], 'movie_companies': ['movie_id', 'company_id', 'company_type_id'], 'aka_name': ['id'], 'movie_info': ['movie_id', 'info_type_id'], 'movie_keyword': ['movie_id', 'keyword_id'], 'person_info': ['info_type_id'], 'comp_cast_type': ['id'], 'complete_cast': ['movie_id', 'subject_id'], 'char_name': ['id'], 'movie_link': ['id', 'link_type_id'], 'company_type': ['id'], 'cast_info': ['movie_id', 'person_role_id', 'role_id', 'person_id'], 'info_type': ['id'], 'company_name': ['id'], 'aka_title': ['movie_id'], 'kind_type': ['id'], 'role_type': ['id'], 'movie_info_idx': ['movie_id'], 'keyword': ['id'], 'link_type': ['id'], 'title': ['id', 'kind_id']},
        'join_root': 'title',
        'join_clauses': ['title.id=aka_title.movie_id', 'title.id=cast_info.movie_id', 'title.id=complete_cast.movie_id', 'title.id=movie_companies.movie_id', 'title.id=movie_info.movie_id', 'title.id=movie_info_idx.movie_id', 'title.id=movie_keyword.movie_id', 'title.id=movie_link.id', 'title.kind_id=kind_type.id', 'cast_info.person_role_id=char_name.id', 'cast_info.role_id=role_type.id', 'cast_info.person_id=aka_name.id', 'cast_info.person_id=name.id', 'complete_cast.subject_id=comp_cast_type.id', 'movie_companies.company_id=company_name.id', 'movie_companies.company_type_id=company_type.id', 'movie_info.info_type_id=info_type.id', 'movie_keyword.keyword_id=keyword.id', 'movie_link.link_type_id=link_type.id', 'info_type.id=person_info.info_type_id'],
        'dataset' : 'imdb',
        'join_root': 'title',
        'join_how': 'outer',
        'use_cols': 'imdb-db',
        'data_dir': 'datasets/job_csv_export/',
    },

    'tpcds-full' : {

        'join_tables': ['item','store_sales',  'store_returns', 'catalog_sales', 'catalog_returns', 'web_sales', 'web_returns', 'inventory', 'promotion', 'date_dim', 'time_dim', 'customer', 'customer_demographics', 'household_demographics', 'customer_address', 'store', 'reason', 'call_center', 'catalog_page', 'ship_mode', 'warehouse', 'web_page', 'web_site', 'income_band'],
        'join_keys': {'store_sales': ['ss_item_sk'], 'item': ['i_item_sk'], 'store_returns': ['sr_item_sk', 'sr_returned_date_sk', 'sr_return_time_sk', 'sr_customer_sk', 'sr_cdemo_sk', 'sr_hdemo_sk', 'sr_addr_sk', 'sr_store_sk', 'sr_reason_sk'], 'catalog_sales': ['cs_item_sk', 'cs_call_center_sk', 'cs_catalog_page_sk', 'cs_ship_mode_sk', 'cs_warehouse_sk'], 'catalog_returns': ['cr_item_sk'], 'web_sales': ['ws_item_sk', 'ws_web_page_sk', 'ws_web_site_sk'], 'web_returns': ['wr_item_sk'], 'inventory': ['inv_item_sk'], 'promotion': ['p_item_sk'], 'date_dim': ['d_date_sk'], 'time_dim': ['t_time_sk'], 'customer': ['c_customer_sk'], 'customer_demographics': ['cd_demo_sk'], 'household_demographics': ['hd_demo_sk', 'hd_income_band_sk'], 'customer_address': ['ca_address_sk'], 'store': ['s_store_sk'], 'reason': ['r_reason_sk'], 'call_center': ['cc_call_center_sk'], 'catalog_page': ['cp_catalog_page_sk'], 'ship_mode': ['sm_ship_mode_sk'], 'warehouse': ['w_warehouse_sk'], 'web_page': ['wp_web_page_sk'], 'web_site': ['web_site_sk'], 'income_band': ['ib_income_band_sk']},
        'join_root': 'item',
        'join_clauses': ['item.i_item_sk=store_returns.sr_item_sk','item.i_item_sk=store_sales.ss_item_sk', 'item.i_item_sk=catalog_sales.cs_item_sk', 'item.i_item_sk=catalog_returns.cr_item_sk', 'item.i_item_sk=web_sales.ws_item_sk', 'item.i_item_sk=web_returns.wr_item_sk', 'item.i_item_sk=inventory.inv_item_sk', 'item.i_item_sk=promotion.p_item_sk', 'store_returns.sr_returned_date_sk=date_dim.d_date_sk', 'store_returns.sr_return_time_sk=time_dim.t_time_sk', 'store_returns.sr_customer_sk=customer.c_customer_sk', 'store_returns.sr_cdemo_sk=customer_demographics.cd_demo_sk', 'store_returns.sr_hdemo_sk=household_demographics.hd_demo_sk', 'store_returns.sr_addr_sk=customer_address.ca_address_sk', 'store_returns.sr_store_sk=store.s_store_sk', 'store_returns.sr_reason_sk=reason.r_reason_sk', 'catalog_sales.cs_call_center_sk=call_center.cc_call_center_sk', 'catalog_sales.cs_catalog_page_sk=catalog_page.cp_catalog_page_sk', 'catalog_sales.cs_ship_mode_sk=ship_mode.sm_ship_mode_sk', 'catalog_sales.cs_warehouse_sk=warehouse.w_warehouse_sk', 'web_sales.ws_web_page_sk=web_page.wp_web_page_sk', 'web_sales.ws_web_site_sk=web_site.web_site_sk', 'household_demographics.hd_income_band_sk=income_band.ib_income_band_sk'],
        'use_cols': 'tpcds-db',
        'join_how': 'outer',
        'dataset': 'tpcds',
        'data_dir': 'datasets/tpcds_2_13_0/',
    },

    'tpcds-benchmark' : {
        'join_tables': ['item','store_sales',  'store_returns', 'catalog_sales', 'catalog_returns', 'web_sales', 'web_returns', 'inventory', 'promotion', 'date_dim', 'time_dim', 'customer', 'customer_demographics', 'household_demographics', 'customer_address', 'store', 'reason', 'call_center', 'catalog_page', 'ship_mode', 'warehouse', 'web_page', 'web_site', 'income_band'],
        'join_keys': {'item': ['i_item_sk'], 'catalog_returns': ['cr_item_sk',  'cr_call_center_sk',  'cr_catalog_page_sk',  'cr_refunded_addr_sk',  'cr_returned_date_sk',  'cr_returning_customer_sk'], 'catalog_sales': ['cs_item_sk',  'cs_bill_cdemo_sk',  'cs_bill_hdemo_sk',  'cs_promo_sk',  'cs_ship_mode_sk',  'cs_sold_time_sk',  'cs_warehouse_sk'], 'inventory': ['inv_item_sk'], 'store_returns': ['sr_item_sk', 'sr_reason_sk', 'sr_store_sk'], 'store_sales': ['ss_item_sk'], 'web_returns': ['wr_item_sk', 'wr_web_page_sk'], 'web_sales': ['ws_item_sk', 'ws_web_site_sk'], 'call_center': ['cc_call_center_sk'], 'catalog_page': ['cp_catalog_page_sk'], 'customer_address': ['ca_address_sk'], 'date_dim': ['d_date_sk'], 'customer': ['c_customer_sk'], 'customer_demographics': ['cd_demo_sk'], 'household_demographics': ['hd_demo_sk', 'hd_income_band_sk'], 'promotion': ['p_promo_sk'], 'ship_mode': ['sm_ship_mode_sk'], 'time_dim': ['t_time_sk'], 'warehouse': ['w_warehouse_sk'], 'reason': ['r_reason_sk'], 'store': ['s_store_sk'], 'web_page': ['wp_web_page_sk'], 'web_site': ['web_site_sk'], 'income_band': ['ib_income_band_sk']},
        'join_root': 'item',
        'join_clauses': ['item.i_item_sk=catalog_returns.cr_item_sk','item.i_item_sk=catalog_sales.cs_item_sk','item.i_item_sk=inventory.inv_item_sk','item.i_item_sk=store_returns.sr_item_sk','item.i_item_sk=store_sales.ss_item_sk','item.i_item_sk=web_returns.wr_item_sk','item.i_item_sk=web_sales.ws_item_sk','catalog_returns.cr_call_center_sk=call_center.cc_call_center_sk','catalog_returns.cr_catalog_page_sk=catalog_page.cp_catalog_page_sk','catalog_returns.cr_refunded_addr_sk=customer_address.ca_address_sk','catalog_returns.cr_returned_date_sk=date_dim.d_date_sk','catalog_returns.cr_returning_customer_sk=customer.c_customer_sk','catalog_sales.cs_bill_cdemo_sk=customer_demographics.cd_demo_sk','catalog_sales.cs_bill_hdemo_sk=household_demographics.hd_demo_sk','catalog_sales.cs_promo_sk=promotion.p_promo_sk','catalog_sales.cs_ship_mode_sk=ship_mode.sm_ship_mode_sk','catalog_sales.cs_sold_time_sk=time_dim.t_time_sk','catalog_sales.cs_warehouse_sk=warehouse.w_warehouse_sk','store_returns.sr_reason_sk=reason.r_reason_sk','store_returns.sr_store_sk=store.s_store_sk','web_returns.wr_web_page_sk=web_page.wp_web_page_sk','web_sales.ws_web_site_sk=web_site.web_site_sk','household_demographics.hd_income_band_sk=income_band.ib_income_band_sk'],
        'use_cols': 'tpcds-benchmark',
        'join_how': 'outer',
        'dataset': 'tpcds',
        'data_dir': 'datasets/tpcds_2_13_0/',
    },
    'syn-multi' : {
        'join_tables': ['table0','table1','table2','table3','table4','table5','table6','table7','table8','table9'],
        'join_keys': {'table0' : ['PK'], 'table1' : ['PK','FK'], 'table2' : ['PK','FK'], 'table3' : ['PK','FK'], 'table4' : ['PK','FK'], 'table5' : ['PK','FK'], 'table6' : ['PK','FK'], 'table7' : ['PK','FK'], 'table8' : ['PK','FK'], 'table9' : ['FK']},
        'join_root': 'table0',
        'join_clauses': ['table0.PK=table1.FK', 'table1.PK=table2.FK', 'table2.PK=table3.FK', 'table3.PK=table4.FK', 'table4.PK=table5.FK', 'table5.PK=table6.FK', 'table6.PK=table7.FK', 'table7.PK=table8.FK', 'table8.PK=table9.FK'],
        'use_cols': 'default',
        'join_how': 'outer',
        'dataset': 'synthetic',
        'data_dir': None,

    }
}
