#!/usr/bin/env python3

from functools import wraps
import os.path

import glog as log
import numpy as np
import pandas as pd

import datasets

CACHE_DIR = "./cache"

TOY_TABLES = {
    "A": pd.DataFrame({"x": [1, 2, 3]}),
    "B": pd.DataFrame({
        "x": [1, 2, 2, 2, 4],
        "y": [10, 20, 20, 30, 30],
        "z": [100, 100, 100, 100, 200],
    }),
    "C": pd.DataFrame({"y": [10, 20, 20, 40]}),
    "D": pd.DataFrame({"z": [100, 100, 200, 300]}),
}
table_dtype ={'aka_name': {'name': object,   'imdb_index': object,   'name_pcode_cf': object,   'name_pcode_nf': object,   'surname_pcode': object,   'md5sum': object},  'aka_title': {'title': object,   'imdb_index': object,   'phonetic_code': object,   'note': object,   'md5sum': object},  'cast_info': {'note': object},  'char_name': {'name': object,   'imdb_index': object,   'name_pcode_nf': object,   'surname_pcode': object,   'md5sum': object},  'comp_cast_type': {'kind': object},  'company_name': {'name': object,   'country_code': object,   'name_pcode_nf': object,   'name_pcode_sf': object,   'md5sum': object},  'company_type': {'kind': object},  'info_type': {'info': object},  'keyword': {'keyword': object, 'phonetic_code': object},  'kind_type': {'kind': object},  'link_type': {'link': object},  'movie_companies': {'note': object},  'movie_info_idx': {'info': object, 'note': object},  'name': {'name': object,   'imdb_index': object,   'gender': object,   'name_pcode_cf': object,   'name_pcode_nf': object,   'surname_pcode': object,   'md5sum': object},  'role_type': {'role': object},  'title': {'title': object,   'imdb_index': object,   'phonetic_code': object,   'series_years': object,   'md5sum': object},  'movie_info': {'info': object, 'note': object},  'person_info': {'info': object, 'note': object},   'customer_address': {'ca_address_id': object,   'ca_street_number': object,   'ca_street_name': object,   'ca_street_type': object,   'ca_suite_number': object,   'ca_city': object,   'ca_county': object,   'ca_state': object,   'ca_zip': object,   'ca_country': object,   'ca_location_type': object},  'customer_demographics': {'cd_gender': object,   'cd_marital_status': object,   'cd_education_status': object,   'cd_credit_rating': object},  'date_dim': {'d_date_id': object,   'd_day_name': object,   'd_quarter_name': object,   'd_holiday': object,   'd_weekend': object,   'd_following_holiday': object,   'd_current_day': object,   'd_current_week': object,   'd_current_month': object,   'd_current_quarter': object,   'd_current_year': object,   'd_date': object},  'warehouse': {'w_warehouse_id': object,   'w_warehouse_name': object,   'w_street_number': object,   'w_street_name': object,   'w_street_type': object,   'w_suite_number': object,   'w_city': object,   'w_county': object,   'w_state': object,   'w_zip': object,   'w_country': object},  'ship_mode': {'sm_ship_mode_id': object,   'sm_type': object,   'sm_code': object,   'sm_carrier': object,   'sm_contract': object},  'time_dim': {'t_time_id': object,   't_am_pm': object,   't_shift': object,   't_sub_shift': object,   't_meal_time': object},  'reason': {'r_reason_id': object, 'r_reason_desc': object},  'item': {'i_item_id': object,   'i_item_desc': object,   'i_brand': object,   'i_class': object,   'i_category': object,   'i_manufact': object,   'i_size': object,   'i_formulation': object,   'i_color': object,   'i_units': object,   'i_container': object,   'i_product_name': object,   'i_rec_start_date': object,   'i_rec_end_date': object},  'store': {'s_store_id': object,   's_store_name': object,   's_hours': object,   's_manager': object,   's_geography_class': object,   's_market_desc': object,   's_market_manager': object,   's_division_name': object,   's_company_name': object,   's_street_number': object,   's_street_name': object,   's_street_type': object,   's_suite_number': object,   's_city': object,   's_county': object,   's_state': object,   's_zip': object,   's_country': object,   's_rec_start_date': object,   's_rec_end_date': object},  'call_center': {'cc_call_center_id': object,   'cc_name': object,   'cc_class': object,   'cc_hours': object,   'cc_manager': object,   'cc_mkt_class': object,   'cc_mkt_desc': object,   'cc_market_manager': object,   'cc_division_name': object,   'cc_company_name': object,   'cc_street_number': object,   'cc_street_name': object,   'cc_street_type': object,   'cc_suite_number': object,   'cc_city': object,   'cc_county': object,   'cc_state': object,   'cc_zip': object,   'cc_country': object,   'cc_rec_start_date': object,   'cc_rec_end_date': object},  'customer': {'c_customer_id': object,   'c_salutation': object,   'c_first_name': object,   'c_last_name': object,   'c_preferred_cust_flag': object,   'c_birth_country': object,   'c_login': object,   'c_email_address': object},  'web_site': {'web_site_id': object,   'web_name': object,   'web_class': object,   'web_manager': object,   'web_mkt_class': object,   'web_mkt_desc': object,   'web_market_manager': object,   'web_company_name': object,   'web_street_number': object,   'web_street_name': object,   'web_street_type': object,   'web_suite_number': object,   'web_city': object,   'web_county': object,   'web_state': object,   'web_zip': object,   'web_country': object,   'web_rec_start_date': object,   'web_rec_end_date': object},  'household_demographics': {'hd_buy_potential': object},  'web_page': {'wp_web_page_id': object,   'wp_autogen_flag': object,   'wp_url': object,   'wp_type': object,   'wp_rec_start_date': object,   'wp_rec_end_date': object},  'promotion': {'p_promo_id': object,   'p_promo_name': object,   'p_channel_dmail': object,   'p_channel_email': object,   'p_channel_catalog': object,   'p_channel_tv': object,   'p_channel_radio': object,   'p_channel_press': object,   'p_channel_event': object,   'p_channel_demo': object,   'p_channel_details': object,   'p_purpose': object,   'p_discount_active': object},  'catalog_page': {'cp_catalog_page_id': object,   'cp_department': object,   'cp_description': object,   'cp_type': object}}

def load(filename, description):
    save_path = os.path.join(CACHE_DIR, datasets.filename_encoder(filename))
    return pd.read_feather(save_path)


def save_result(filename, subdir=None, description="result"):

    def decorator(func):

        @wraps(func)
        def wrapper(*fargs, **kwargs):
            os.makedirs(CACHE_DIR, exist_ok=True)
            if subdir is not None:
                os.makedirs(os.path.join(CACHE_DIR, subdir), exist_ok=True)
                save_path = os.path.join(CACHE_DIR, subdir, datasets.filename_encoder(filename))
            else:
                save_path = os.path.join(CACHE_DIR, datasets.filename_encoder(filename))
            if os.path.exists(save_path):
                ret = pd.read_feather(save_path)
            else:
                ret = func(*fargs, **kwargs)
                ret.to_feather(save_path)
            return ret

        return wrapper

    return decorator


def load_table(table, data_dir, dataset , **kwargs):

    usecols = kwargs.get("usecols")
    if usecols is not None:
        usecols = datasets.get_use_column(dataset,table,usecols)
    kwargs.update({"usecols": usecols})
    if usecols is None:
        usecols = ["ALL"]
    @save_result("{}-{}.df".format(table, "-".join(usecols)),
                 description=f"dataframe of `{table}`",subdir=data_dir)
    def work():
        print(table, kwargs)
        print(f"load {table} from {data_dir} ")
        if table in table_dtype.keys():
            dtype_dict = table_dtype[table]
        else:
            dtype_dict = dict()
        key_dtype = kwargs.get("dtype")
        dtype = dict(dtype_dict,**key_dtype)
        kwargs.update({"dtype": dtype})

        df = pd.read_csv(os.path.join(data_dir, f"{table}.csv"),
                           low_memory=False,keep_default_na=False,na_values=[''],
                           **kwargs)
        print(f'Data type : {table} - {dtype_dict}\n{df.dtypes}')
        for col in df.columns:
            if df[col].dtype == object :
                df[col] = df[col].str.strip()
        return df

    return work()
