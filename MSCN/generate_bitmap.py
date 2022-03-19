import argparse
import time
import os
from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
import collections
import sys

from mscn.util_common import *
from mscn.data_common import *

NULL = -1

table_dtype = {'aka_name': {'name': object, 'imdb_index': object, 'name_pcode_cf': object, 'name_pcode_nf': object, 'surname_pcode': object, 'md5sum': object}, 'aka_title': {'title': object, 'imdb_index': object, 'phonetic_code': object, 'note': object, 'md5sum': object}, 'cast_info': {'note': object}, 'char_name': {'name': object, 'imdb_index': object, 'name_pcode_nf': object, 'surname_pcode': object, 'md5sum': object}, 'comp_cast_type': {'kind': object}, 'company_name': {'name': object, 'country_code': object, 'name_pcode_nf': object, 'name_pcode_sf': object, 'md5sum': object}, 'company_type': {'kind': object}, 'complete_cast': {}, 'info_type': {'info': object}, 'keyword': {'keyword': object, 'phonetic_code': object}, 'kind_type': {'kind': object}, 'link_type': {'link': object}, 'movie_companies': {'note': object}, 'movie_info_idx': {'info': object, 'note': object}, 'movie_keyword': {}, 'movie_link': {}, 'name': {'name': object, 'imdb_index': object, 'gender': object, 'name_pcode_cf': object, 'name_pcode_nf': object, 'surname_pcode': object, 'md5sum': object}, 'role_type': {'role': object}, 'title': {'title': object, 'imdb_index': object, 'phonetic_code': object, 'series_years': object, 'md5sum': object}, 'movie_info': {'info': object, 'note': object}, 'person_info': {'info': object, 'note': object}, 'customer_address': {'ca_address_id': object, 'ca_street_number': object, 'ca_street_name': object, 'ca_street_type': object, 'ca_suite_number': object, 'ca_city': object, 'ca_county': object, 'ca_state': object, 'ca_zip': object, 'ca_country': object, 'ca_location_type': object}, 'customer_demographics': {'cd_gender': object, 'cd_marital_status': object, 'cd_education_status': object, 'cd_credit_rating': object}, 'date_dim': {'d_date_id': object, 'd_day_name': object, 'd_quarter_name': object, 'd_holiday': object, 'd_weekend': object, 'd_following_holiday': object, 'd_current_day': object, 'd_current_week': object, 'd_current_month': object, 'd_current_quarter': object, 'd_current_year': object, 'd_date': object}, 'warehouse': {'w_warehouse_id': object, 'w_warehouse_name': object, 'w_street_number': object, 'w_street_name': object, 'w_street_type': object, 'w_suite_number': object, 'w_city': object, 'w_county': object, 'w_state': object, 'w_zip': object, 'w_country': object}, 'ship_mode': {'sm_ship_mode_id': object, 'sm_type': object, 'sm_code': object, 'sm_carrier': object, 'sm_contract': object}, 'time_dim': {'t_time_id': object, 't_am_pm': object, 't_shift': object, 't_sub_shift': object, 't_meal_time': object}, 'reason': {'r_reason_id': object, 'r_reason_desc': object}, 'income_band': {}, 'item': {'i_item_id': object, 'i_item_desc': object, 'i_brand': object, 'i_class': object, 'i_category': object, 'i_manufact': object, 'i_size': object, 'i_formulation': object, 'i_color': object, 'i_units': object, 'i_container': object, 'i_product_name': object, 'i_rec_start_date': object, 'i_rec_end_date': object}, 'store': {'s_store_id': object, 's_store_name': object, 's_hours': object, 's_manager': object, 's_geography_class': object, 's_market_desc': object, 's_market_manager': object, 's_division_name': object, 's_company_name': object, 's_street_number': object, 's_street_name': object, 's_street_type': object, 's_suite_number': object, 's_city': object, 's_county': object, 's_state': object, 's_zip': object, 's_country': object, 's_rec_start_date': object, 's_rec_end_date': object}, 'call_center': {'cc_call_center_id': object, 'cc_name': object, 'cc_class': object, 'cc_hours': object, 'cc_manager': object, 'cc_mkt_class': object, 'cc_mkt_desc': object, 'cc_market_manager': object, 'cc_division_name': object, 'cc_company_name': object, 'cc_street_number': object, 'cc_street_name': object, 'cc_street_type': object, 'cc_suite_number': object, 'cc_city': object, 'cc_county': object, 'cc_state': object, 'cc_zip': object, 'cc_country': object, 'cc_rec_start_date': object, 'cc_rec_end_date': object}, 'customer': {'c_customer_id': object, 'c_salutation': object, 'c_first_name': object, 'c_last_name': object, 'c_preferred_cust_flag': object, 'c_birth_country': object, 'c_login': object, 'c_email_address': object}, 'web_site': {'web_site_id': object, 'web_name': object, 'web_class': object, 'web_manager': object, 'web_mkt_class': object, 'web_mkt_desc': object, 'web_market_manager': object, 'web_company_name': object, 'web_street_number': object, 'web_street_name': object, 'web_street_type': object, 'web_suite_number': object, 'web_city': object, 'web_county': object, 'web_state': object, 'web_zip': object, 'web_country': object, 'web_rec_start_date': object, 'web_rec_end_date': object}, 'store_returns': {}, 'household_demographics': {'hd_buy_potential': object}, 'web_page': {'wp_web_page_id': object, 'wp_autogen_flag': object, 'wp_url': object, 'wp_type': object, 'wp_rec_start_date': object, 'wp_rec_end_date': object}, 'promotion': {'p_promo_id': object, 'p_promo_name': object, 'p_channel_dmail': object, 'p_channel_email': object, 'p_channel_catalog': object, 'p_channel_tv': object, 'p_channel_radio': object, 'p_channel_press': object, 'p_channel_event': object, 'p_channel_demo': object, 'p_channel_details': object, 'p_purpose': object, 'p_discount_active': object}, 'catalog_page': {'cp_catalog_page_id': object, 'cp_department': object, 'cp_description': object, 'cp_type': object}, 'inventory': {}, 'catalog_returns': {}, 'web_returns': {}, 'web_sales': {}, 'catalog_sales': {}, 'store_sales': {}}

TPCDS_ALIAS_DICT= {'ss': 'store_sales','sr': 'store_returns','cs': 'catalog_sales','cr': 'catalog_returns','ws': 'web_sales','wr': 'web_returns','inv': 'inventory','s': 'store','cc': 'call_center','cp': 'catalog_page','web': 'web_site','wp': 'web_page','w': 'warehouse','c': 'customer','ca': 'customer_address','cd': 'customer_demographics','d': 'date_dim','hd': 'household_demographics','i': 'item','ib': 'income_band','p': 'promotion','r': 'reason','sm': 'ship_mode','t': 'time_dim'}
JOB_ALIAS_DICT = { 'n':'name',  'mc':'movie_companies',  'an':'aka_name',  'mi':'movie_info',  'mk':'movie_keyword',  'pi':'person_info',  'cct':'comp_cast_type',  'cc':'complete_cast',  'ch_n':'char_name',
 'ml':'movie_link',  'ct':'company_type',  'ci':'cast_info',  'it':'info_type',  'cn':'company_name',  'at':'aka_title',  'kt':'kind_type',  'rt':'role_type',  'mi_idx':'movie_info_idx',  'k':'keyword',  'lt':'link_type',  't':'title'}

TPCDS_TABLES = ['store_sales','store_returns','catalog_sales','catalog_returns','web_sales','web_returns','inventory','store','call_center','catalog_page','web_site','web_page','warehouse','customer','customer_address','customer_demographics','date_dim','household_demographics','item','income_band','promotion','reason','ship_mode','time_dim']
JOB_TABLES = ['name', 'movie_companies', 'aka_name', 'movie_info', 'movie_keyword', 'person_info', 'comp_cast_type', 'complete_cast', 'char_name',  'movie_link', 'company_type', 'cast_info', 'info_type', 'company_name', 'aka_title', 'kind_type', 'role_type', 'movie_info_idx', 'keyword', 'link_type', 'title']

JoinSpec = collections.namedtuple("JoinSpec", [
    "join_tables", "join_keys", "join_clauses", "join_graph", "join_tree",
    "join_root", "join_how", "join_name"
])

imdbDtypeDict = {'aka_name.id' : 'int', 'aka_name.person_id' : 'int', 'aka_name.name' : 'str', 'aka_name.imdb_index' : 'str', 'aka_name.name_pcode_cf' : 'str', 'aka_name.name_pcode_nf' : 'str', 'aka_name.surname_pcode' : 'str', 'aka_name.md5sum' : 'str', 'aka_title.id' : 'int', 'aka_title.movie_id' : 'int', 'aka_title.title' : 'str', 'aka_title.imdb_index' : 'str', 'aka_title.kind_id' : 'int', 'aka_title.production_year' : 'int', 'aka_title.phonetic_code' : 'str', 'aka_title.episode_of_id' : 'int', 'aka_title.season_nr' : 'int', 'aka_title.episode_nr' : 'int', 'aka_title.note' : 'str', 'aka_title.md5sum' : 'str', 'cast_info.id' : 'int', 'cast_info.person_id' : 'int', 'cast_info.movie_id' : 'int', 'cast_info.person_role_id' : 'int', 'cast_info.note' : 'str', 'cast_info.nr_order' : 'int', 'cast_info.role_id' : 'int', 'char_name.id' : 'int', 'char_name.name' : 'str', 'char_name.imdb_index' : 'str', 'char_name.imdb_id' : 'int', 'char_name.name_pcode_nf' : 'str', 'char_name.surname_pcode' : 'str', 'char_name.md5sum' : 'str', 'comp_cast_type.id' : 'int', 'comp_cast_type.kind' : 'str', 'company_name.id' : 'int', 'company_name.name' : 'str', 'company_name.country_code' : 'str', 'company_name.imdb_id' : 'int', 'company_name.name_pcode_nf' : 'str', 'company_name.name_pcode_sf' : 'str', 'company_name.md5sum' : 'str', 'company_type.id' : 'int', 'company_type.kind' : 'str', 'complete_cast.id' : 'int', 'complete_cast.movie_id' : 'int', 'complete_cast.subject_id' : 'int', 'complete_cast.status_id' : 'int', 'info_type.id' : 'int', 'info_type.info' : 'str', 'keyword.id' : 'int', 'keyword.keyword' : 'str', 'keyword.phonetic_code' : 'str', 'kind_type.id' : 'int', 'kind_type.kind' : 'str', 'link_type.id' : 'int', 'link_type.link' : 'str', 'movie_companies.id' : 'int', 'movie_companies.movie_id' : 'int', 'movie_companies.company_id' : 'int', 'movie_companies.company_type_id' : 'int', 'movie_companies.note' : 'str', 'movie_info_idx.id' : 'int', 'movie_info_idx.movie_id' : 'int', 'movie_info_idx.info_type_id' : 'int', 'movie_info_idx.info' : 'str', 'movie_info_idx.note' : 'str', 'movie_keyword.id' : 'int', 'movie_keyword.movie_id' : 'int', 'movie_keyword.keyword_id' : 'int', 'movie_link.id' : 'int', 'movie_link.movie_id' : 'int', 'movie_link.linked_movie_id' : 'int', 'movie_link.link_type_id' : 'int', 'name.id' : 'int', 'name.name' : 'str', 'name.imdb_index' : 'str', 'name.imdb_id' : 'int', 'name.gender' : 'str', 'name.name_pcode_cf' : 'str', 'name.name_pcode_nf' : 'str', 'name.surname_pcode' : 'str', 'name.md5sum' : 'str', 'role_type.id' : 'int', 'role_type.role' : 'str', 'title.id' : 'int', 'title.title' : 'str', 'title.imdb_index' : 'str', 'title.kind_id' : 'int', 'title.production_year' : 'int', 'title.imdb_id' : 'int', 'title.phonetic_code' : 'str', 'title.episode_of_id' : 'int', 'title.season_nr' : 'int', 'title.episode_nr' : 'int', 'title.series_years' : 'str', 'title.md5sum' : 'str', 'movie_info.id' : 'int', 'movie_info.movie_id' : 'int', 'movie_info.info_type_id' : 'int', 'movie_info.info' : 'str', 'movie_info.note' : 'str', 'person_info.id' : 'int', 'person_info.person_id' : 'int', 'person_info.info_type_id' : 'int', 'person_info.info' : 'str', 'person_info.note' : 'str', }
tpcdsDtypeDict = {'customer_address.ca_address_sk': 'int',  'customer_address.ca_address_id': 'str',  'customer_address.ca_street_number': 'str',     'customer_address.ca_street_name': 'str',  'customer_address.ca_street_type': 'str',  'customer_address.ca_suite_number': 'str',  'customer_address.ca_city': 'str',     'customer_address.ca_county': 'str',  'customer_address.ca_state': 'str',  'customer_address.ca_zip': 'str',  'customer_address.ca_country': 'str',     'customer_address.ca_gmt_offset': 'float',  'customer_address.ca_location_type': 'str',  'customer_demographics.cd_demo_sk': 'int',  'customer_demographics.cd_gender': 'str',     'customer_demographics.cd_marital_status': 'str',  'customer_demographics.cd_education_status': 'str',  'customer_demographics.cd_purchase_estimate': 'int',     'customer_demographics.cd_credit_rating': 'str',  'customer_demographics.cd_dep_count': 'int',  'customer_demographics.cd_dep_employed_count': 'int',     'customer_demographics.cd_dep_college_count': 'int',  'date_dim.d_date_sk': 'int',  'date_dim.d_date_id': 'str',     'date_dim.d_month_seq': 'int',  'date_dim.d_week_seq': 'int',  'date_dim.d_quarter_seq': 'int',     'date_dim.d_year': 'int',  'date_dim.d_dow': 'int',  'date_dim.d_moy': 'int',  'date_dim.d_dom': 'int',     'date_dim.d_qoy': 'int',  'date_dim.d_fy_year': 'int',  'date_dim.d_fy_quarter_seq': 'int',     'date_dim.d_fy_week_seq': 'int',  'date_dim.d_day_name': 'str',  'date_dim.d_quarter_name': 'str',  'date_dim.d_holiday': 'str',     'date_dim.d_weekend': 'str',  'date_dim.d_following_holiday': 'str',  'date_dim.d_first_dom': 'int',     'date_dim.d_last_dom': 'int',  'date_dim.d_same_day_ly': 'int',  'date_dim.d_same_day_lq': 'int',     'date_dim.d_current_day': 'str',  'date_dim.d_current_week': 'str',  'date_dim.d_current_month': 'str',     'date_dim.d_current_quarter': 'str',  'date_dim.d_current_year': 'str',  'warehouse.w_warehouse_sk': 'int',     'warehouse.w_warehouse_id': 'str',  'warehouse.w_warehouse_name': 'str',  'warehouse.w_warehouse_sq_ft': 'int',     'warehouse.w_street_number': 'str',  'warehouse.w_street_name': 'str',  'warehouse.w_street_type': 'str',     'warehouse.w_suite_number': 'str',  'warehouse.w_city': 'str',  'warehouse.w_county': 'str',  'warehouse.w_state': 'str',     'warehouse.w_zip': 'str',  'warehouse.w_country': 'str',  'warehouse.w_gmt_offset': 'float',  'ship_mode.sm_ship_mode_sk': 'int',     'ship_mode.sm_ship_mode_id': 'str',  'ship_mode.sm_type': 'str',  'ship_mode.sm_code': 'str',  'ship_mode.sm_carrier': 'str',     'ship_mode.sm_contract': 'str',  'time_dim.t_time_sk': 'int',  'time_dim.t_time_id': 'str',  'time_dim.t_time': 'int',     'time_dim.t_hour': 'int',  'time_dim.t_minute': 'int',  'time_dim.t_second': 'int',  'time_dim.t_am_pm': 'str',     'time_dim.t_shift': 'str',  'time_dim.t_sub_shift': 'str',  'time_dim.t_meal_time': 'str',  'reason.r_reason_sk': 'int',     'reason.r_reason_id': 'str',  'reason.r_reason_desc': 'str',  'income_band.ib_income_band_sk': 'int',     'income_band.ib_lower_bound': 'int',  'income_band.ib_upper_bound': 'int',  'item.i_item_sk': 'int',     'item.i_item_id': 'str',  'item.i_item_desc': 'str',  'item.i_current_price': 'float',  'item.i_wholesale_cost': 'float',     'item.i_brand_id': 'int',  'item.i_brand': 'str',  'item.i_class_id': 'int',  'item.i_class': 'str',     'item.i_category_id': 'int',  'item.i_category': 'str',  'item.i_manufact_id': 'int',     'item.i_manufact': 'str',  'item.i_size': 'str',  'item.i_formulation': 'str',  'item.i_color': 'str',     'item.i_units': 'str',  'item.i_container': 'str',  'item.i_manager_id': 'int',  'item.i_product_name': 'str',     'store.s_store_sk': 'int',  'store.s_store_id': 'str',  'store.s_closed_date_sk': 'int',     'store.s_store_name': 'str',  'store.s_number_employees': 'int',  'store.s_floor_space': 'int',     'store.s_hours': 'str',  'store.s_manager': 'str',  'store.s_market_id': 'int',  'store.s_geography_class': 'str',     'store.s_market_desc': 'str',  'store.s_market_manager': 'str',  'store.s_division_id': 'int',     'store.s_division_name': 'str',  'store.s_company_id': 'int',  'store.s_company_name': 'str',     'store.s_street_number': 'str',  'store.s_street_name': 'str',  'store.s_street_type': 'str',     'store.s_suite_number': 'str',  'store.s_city': 'str',  'store.s_county': 'str',  'store.s_state': 'str',     'store.s_zip': 'str',  'store.s_country': 'str',  'store.s_gmt_offset': 'float',  'store.s_tax_precentage': 'float',     'call_center.cc_call_center_sk': 'int',  'call_center.cc_call_center_id': 'str',  'call_center.cc_closed_date_sk': 'int',     'call_center.cc_open_date_sk': 'int',  'call_center.cc_name': 'str',  'call_center.cc_class': 'str',  'call_center.cc_employees': 'int',     'call_center.cc_sq_ft': 'int',  'call_center.cc_hours': 'str',  'call_center.cc_manager': 'str',  'call_center.cc_mkt_id': 'int',     'call_center.cc_mkt_class': 'str',  'call_center.cc_mkt_desc': 'str',  'call_center.cc_market_manager': 'str',     'call_center.cc_division': 'int',  'call_center.cc_division_name': 'str',  'call_center.cc_company': 'int',     'call_center.cc_company_name': 'str',  'call_center.cc_street_number': 'str',  'call_center.cc_street_name': 'str',     'call_center.cc_street_type': 'str',  'call_center.cc_suite_number': 'str',  'call_center.cc_city': 'str',  'call_center.cc_county': 'str',     'call_center.cc_state': 'str',  'call_center.cc_zip': 'str',  'call_center.cc_country': 'str',  'call_center.cc_gmt_offset': 'float',     'call_center.cc_tax_percentage': 'float',  'customer.c_customer_sk': 'int',  'customer.c_customer_id': 'str',     'customer.c_current_cdemo_sk': 'int',  'customer.c_current_hdemo_sk': 'int',  'customer.c_current_addr_sk': 'int',     'customer.c_first_shipto_date_sk': 'int',  'customer.c_first_sales_date_sk': 'int',  'customer.c_salutation': 'str',     'customer.c_first_name': 'str',  'customer.c_last_name': 'str',  'customer.c_preferred_cust_flag': 'str',     'customer.c_birth_day': 'int',  'customer.c_birth_month': 'int',  'customer.c_birth_year': 'int',     'customer.c_birth_country': 'str',  'customer.c_login': 'str',  'customer.c_email_address': 'str',     'customer.c_last_review_date_sk': 'int',  'web_site.web_site_sk': 'int',  'web_site.web_site_id': 'str',     'web_site.web_name': 'str',  'web_site.web_open_date_sk': 'int',  'web_site.web_close_date_sk': 'int',     'web_site.web_class': 'str',  'web_site.web_manager': 'str',  'web_site.web_mkt_id': 'int',  'web_site.web_mkt_class': 'str',     'web_site.web_mkt_desc': 'str',  'web_site.web_market_manager': 'str',  'web_site.web_company_id': 'int',     'web_site.web_company_name': 'str',  'web_site.web_street_number': 'str',  'web_site.web_street_name': 'str',     'web_site.web_street_type': 'str',  'web_site.web_suite_number': 'str',  'web_site.web_city': 'str',  'web_site.web_county': 'str',     'web_site.web_state': 'str',  'web_site.web_zip': 'str',  'web_site.web_country': 'str',  'web_site.web_gmt_offset': 'float',     'web_site.web_tax_percentage': 'float',  'store_returns.sr_returned_date_sk': 'int',  'store_returns.sr_return_time_sk': 'int',     'store_returns.sr_item_sk': 'int',  'store_returns.sr_customer_sk': 'int',  'store_returns.sr_cdemo_sk': 'int',     'store_returns.sr_hdemo_sk': 'int',  'store_returns.sr_addr_sk': 'int',  'store_returns.sr_store_sk': 'int',     'store_returns.sr_reason_sk': 'int',  'store_returns.sr_ticket_number': 'int',  'store_returns.sr_return_quantity': 'int',     'store_returns.sr_return_amt': 'float',  'store_returns.sr_return_tax': 'float',  'store_returns.sr_return_amt_inc_tax': 'float',  'store_returns.sr_fee': 'float',     'store_returns.sr_return_ship_cost': 'float',  'store_returns.sr_refunded_cash': 'float',  'store_returns.sr_reversed_charge': 'float',     'store_returns.sr_store_credit': 'float',  'store_returns.sr_net_loss': 'float',  'household_demographics.hd_demo_sk': 'int',     'household_demographics.hd_income_band_sk': 'int',  'household_demographics.hd_buy_potential': 'str',  'household_demographics.hd_dep_count': 'int',     'household_demographics.hd_vehicle_count': 'int',  'web_page.wp_web_page_sk': 'int',  'web_page.wp_web_page_id': 'str',     'web_page.wp_creation_date_sk': 'int',  'web_page.wp_access_date_sk': 'int',  'web_page.wp_autogen_flag': 'str',     'web_page.wp_customer_sk': 'int',  'web_page.wp_url': 'str',  'web_page.wp_type': 'str',  'web_page.wp_char_count': 'int',     'web_page.wp_link_count': 'int',  'web_page.wp_image_count': 'int',  'web_page.wp_max_ad_count': 'int',     'promotion.p_promo_sk': 'int',  'promotion.p_promo_id': 'str',  'promotion.p_start_date_sk': 'int',     'promotion.p_end_date_sk': 'int',  'promotion.p_item_sk': 'int',  'promotion.p_cost': 'float',     'promotion.p_response_target': 'int',  'promotion.p_promo_name': 'str',  'promotion.p_channel_dmail': 'str',     'promotion.p_channel_email': 'str',  'promotion.p_channel_catalog': 'str',  'promotion.p_channel_tv': 'str',     'promotion.p_channel_radio': 'str',  'promotion.p_channel_press': 'str',  'promotion.p_channel_event': 'str',     'promotion.p_channel_demo': 'str',  'promotion.p_channel_details': 'str',  'promotion.p_purpose': 'str',     'promotion.p_discount_active': 'str',  'catalog_page.cp_catalog_page_sk': 'int',  'catalog_page.cp_catalog_page_id': 'str',     'catalog_page.cp_start_date_sk': 'int',  'catalog_page.cp_end_date_sk': 'int',  'catalog_page.cp_department': 'str',     'catalog_page.cp_catalog_number': 'int',  'catalog_page.cp_catalog_page_number': 'int',  'catalog_page.cp_description': 'str',     'catalog_page.cp_type': 'str',  'inventory.inv_date_sk': 'int',  'inventory.inv_item_sk': 'int',     'inventory.inv_warehouse_sk': 'int',  'inventory.inv_quantity_on_hand': 'int',     'catalog_returns.cr_returned_date_sk': 'int',  'catalog_returns.cr_returned_time_sk': 'int',  'catalog_returns.cr_item_sk': 'int',     'catalog_returns.cr_refunded_customer_sk': 'int',  'catalog_returns.cr_refunded_cdemo_sk': 'int',     'catalog_returns.cr_refunded_hdemo_sk': 'int',  'catalog_returns.cr_refunded_addr_sk': 'int',     'catalog_returns.cr_returning_customer_sk': 'int',  'catalog_returns.cr_returning_cdemo_sk': 'int',     'catalog_returns.cr_returning_hdemo_sk': 'int',  'catalog_returns.cr_returning_addr_sk': 'int',     'catalog_returns.cr_call_center_sk': 'int',  'catalog_returns.cr_catalog_page_sk': 'int',  'catalog_returns.cr_ship_mode_sk': 'int',     'catalog_returns.cr_warehouse_sk': 'int',  'catalog_returns.cr_reason_sk': 'int',  'catalog_returns.cr_order_number': 'int',     'catalog_returns.cr_return_quantity': 'int',  'catalog_returns.cr_return_amount': 'float',  'catalog_returns.cr_return_tax': 'float',     'catalog_returns.cr_return_amt_inc_tax': 'float',  'catalog_returns.cr_fee': 'float',  'catalog_returns.cr_return_ship_cost': 'float',     'catalog_returns.cr_refunded_cash': 'float',  'catalog_returns.cr_reversed_charge': 'float',  'catalog_returns.cr_store_credit': 'float',     'catalog_returns.cr_net_loss': 'float',  'web_returns.wr_returned_date_sk': 'int',  'web_returns.wr_returned_time_sk': 'int',     'web_returns.wr_item_sk': 'int',  'web_returns.wr_refunded_customer_sk': 'int',  'web_returns.wr_refunded_cdemo_sk': 'int',     'web_returns.wr_refunded_hdemo_sk': 'int',  'web_returns.wr_refunded_addr_sk': 'int',     'web_returns.wr_returning_customer_sk': 'int',  'web_returns.wr_returning_cdemo_sk': 'int',     'web_returns.wr_returning_hdemo_sk': 'int',  'web_returns.wr_returning_addr_sk': 'int',     'web_returns.wr_web_page_sk': 'int',  'web_returns.wr_reason_sk': 'int',  'web_returns.wr_order_number': 'int',     'web_returns.wr_return_quantity': 'int',  'web_returns.wr_return_amt': 'float',  'web_returns.wr_return_tax': 'float',     'web_returns.wr_return_amt_inc_tax': 'float',  'web_returns.wr_fee': 'float',  'web_returns.wr_return_ship_cost': 'float',     'web_returns.wr_refunded_cash': 'float',  'web_returns.wr_reversed_charge': 'float',  'web_returns.wr_account_credit': 'float',     'web_returns.wr_net_loss': 'float',  'web_sales.ws_sold_date_sk': 'int',  'web_sales.ws_sold_time_sk': 'int',     'web_sales.ws_ship_date_sk': 'int',  'web_sales.ws_item_sk': 'int',  'web_sales.ws_bill_customer_sk': 'int',     'web_sales.ws_bill_cdemo_sk': 'int',  'web_sales.ws_bill_hdemo_sk': 'int',  'web_sales.ws_bill_addr_sk': 'int',     'web_sales.ws_ship_customer_sk': 'int',  'web_sales.ws_ship_cdemo_sk': 'int',  'web_sales.ws_ship_hdemo_sk': 'int',     'web_sales.ws_ship_addr_sk': 'int',  'web_sales.ws_web_page_sk': 'int',  'web_sales.ws_web_site_sk': 'int',     'web_sales.ws_ship_mode_sk': 'int',  'web_sales.ws_warehouse_sk': 'int',  'web_sales.ws_promo_sk': 'int',     'web_sales.ws_order_number': 'int',  'web_sales.ws_quantity': 'int',  'web_sales.ws_wholesale_cost': 'float',     'web_sales.ws_list_price': 'float',  'web_sales.ws_sales_price': 'float',  'web_sales.ws_ext_discount_amt': 'float',     'web_sales.ws_ext_sales_price': 'float',  'web_sales.ws_ext_wholesale_cost': 'float',  'web_sales.ws_ext_list_price': 'float',     'web_sales.ws_ext_tax': 'float',  'web_sales.ws_coupon_amt': 'float',  'web_sales.ws_ext_ship_cost': 'float',  'web_sales.ws_net_paid': 'float',     'web_sales.ws_net_paid_inc_tax': 'float',  'web_sales.ws_net_paid_inc_ship': 'float',  'web_sales.ws_net_paid_inc_ship_tax': 'float',     'web_sales.ws_net_profit': 'float',  'catalog_sales.cs_sold_date_sk': 'int',  'catalog_sales.cs_sold_time_sk': 'int',     'catalog_sales.cs_ship_date_sk': 'int',  'catalog_sales.cs_bill_customer_sk': 'int',  'catalog_sales.cs_bill_cdemo_sk': 'int',     'catalog_sales.cs_bill_hdemo_sk': 'int',  'catalog_sales.cs_bill_addr_sk': 'int',  'catalog_sales.cs_ship_customer_sk': 'int',     'catalog_sales.cs_ship_cdemo_sk': 'int',  'catalog_sales.cs_ship_hdemo_sk': 'int',  'catalog_sales.cs_ship_addr_sk': 'int',     'catalog_sales.cs_call_center_sk': 'int',  'catalog_sales.cs_catalog_page_sk': 'int',  'catalog_sales.cs_ship_mode_sk': 'int',     'catalog_sales.cs_warehouse_sk': 'int',  'catalog_sales.cs_item_sk': 'int',  'catalog_sales.cs_promo_sk': 'int',     'catalog_sales.cs_order_number': 'int',  'catalog_sales.cs_quantity': 'int',  'catalog_sales.cs_wholesale_cost': 'float',     'catalog_sales.cs_list_price': 'float',  'catalog_sales.cs_sales_price': 'float',  'catalog_sales.cs_ext_discount_amt': 'float',     'catalog_sales.cs_ext_sales_price': 'float',  'catalog_sales.cs_ext_wholesale_cost': 'float',  'catalog_sales.cs_ext_list_price': 'float',     'catalog_sales.cs_ext_tax': 'float',  'catalog_sales.cs_coupon_amt': 'float',  'catalog_sales.cs_ext_ship_cost': 'float',  'catalog_sales.cs_net_paid': 'float',     'catalog_sales.cs_net_paid_inc_tax': 'float',  'catalog_sales.cs_net_paid_inc_ship': 'float',  'catalog_sales.cs_net_paid_inc_ship_tax': 'float',     'catalog_sales.cs_net_profit': 'float',  'store_sales.ss_sold_date_sk': 'int',  'store_sales.ss_sold_time_sk': 'int',     'store_sales.ss_item_sk': 'int',  'store_sales.ss_customer_sk': 'int',  'store_sales.ss_cdemo_sk': 'int',     'store_sales.ss_hdemo_sk': 'int',  'store_sales.ss_addr_sk': 'int',  'store_sales.ss_store_sk': 'int',     'store_sales.ss_promo_sk': 'int',  'store_sales.ss_ticket_number': 'int',  'store_sales.ss_quantity': 'int',     'store_sales.ss_wholesale_cost': 'float',  'store_sales.ss_list_price': 'float',  'store_sales.ss_sales_price': 'float',     'store_sales.ss_ext_discount_amt': 'float',  'store_sales.ss_ext_sales_price': 'float',  'store_sales.ss_ext_wholesale_cost': 'float',     'store_sales.ss_ext_list_price': 'float',  'store_sales.ss_ext_tax': 'float',  'store_sales.ss_coupon_amt': 'float',  'store_sales.ss_net_paid': 'float',     'store_sales.ss_net_paid_inc_tax': 'float',  'store_sales.ss_net_profit': 'float', 'item.i_rec_start_date': 'date',  'item.i_rec_end_date': 'date',  'date_dim.d_date': 'date',  'store.s_rec_start_date': 'date',  'store.s_rec_end_date': 'date',  'call_center.cc_rec_start_date': 'date',  'call_center.cc_rec_end_date': 'date',  'web_page.wp_rec_start_date': 'date',  'web_page.wp_rec_end_date': 'date',  'web_site.web_rec_start_date': 'date',  'web_site.web_rec_end_date': 'date'}


SYN_S_TABLES  = ['table0']
SYN_S_ALIAS_DICT = {'t0':'table0'}
SYM_S_DtypeDict = {'table0.col0':'int', 'table0.col1':'int', 'table0.col2':'int', 'table0.col3':'int', 'table0.col4':'int', 'table0.col5':'int', 'table0.col6':'int', 'table0.col7':'int', 'table0.col8':'int', 'table0.col9':'int' }

SYN_M_TABLES = ['table0', 'table1', 'table2', 'table3', 'table4', 'table5', 'table6', 'table7', 'table8', 'table9']
SYN_M_ALIAS_DICT = {'t0' : 'table0', 't1' : 'table1', 't2' : 'table2', 't3' : 'table3', 't4' : 'table4', 't5' : 'table5', 't6' : 'table6', 't7' : 'table7', 't8' : 'table8', 't9' : 'table9'}
SYN_M_DtypeDict = {'table0.PK' : 'int', 'table1.PK' : 'int', 'table1.FK' : 'int', 'table2.PK' : 'int', 'table2.FK' : 'int', 'table3.PK' : 'int', 'table3.FK' : 'int', 'table4.PK' : 'int', 'table4.FK' : 'int', 'table5.PK' : 'int', 'table5.FK' : 'int', 'table6.PK' : 'int', 'table6.FK' : 'int', 'table7.PK' : 'int', 'table7.FK' : 'int', 'table8.PK' : 'int', 'table8.FK' : 'int', 'table9.PK' : 'int', 'table9.FK' : 'int'}


def get_query_pred(col_type,c,o,v):
    if o == "IS_NULL":
        pred = f"{c}.isnull()"
    elif o == "IS_NOT_NULL":
        pred = f"{c}.notnull()"
    elif o == 'LIKE':
        special_char  = ['^','$','.','?','*','+','(',')','[',']','{','}']
        for s_char in special_char:
            if s_char in v:
                v = v.replace(s_char,f"\{s_char}")
        v = v.replace('\\%', '.*').replace('%', '.*')
        pred = f"""{c}.str.match("{v}",na=False)"""
    elif o == 'NOT_LIKE':
        special_char  = ['^','$','.','?','*','+','(',')','[',']','{','}']
        for s_char in special_char:
            if s_char in v:
                v = v.replace(s_char,f"\{s_char}")
        v = v.replace('\\%', '.*').replace('%', '.*')
        pred = f"""not ( {c}.str.match("{v}",na=True) )"""
    elif o == 'IN':
        # if type(v) == str:
            # v = f"('{v}')"
        pred = f" {c} in {v}"
    elif o == 'NOT_IN':
        # if type(v) == str:
            # v = f"('{v}')"
        pred = f" {c} not in {v} and {c}.notnull()"
    elif o == '!=':
        if col_type =='str' or col_type == 'date':
            v = f""" "{v}" """
        if col_type == 'int':
            v = int(v)
        pred = f"{c} {o} {v} and {c}.notnull() "
    elif o in ['>=','>','=','<','<='] :
        if o == '=' :
            o = '=='
        if col_type =='str' or col_type == 'date':
            v = f""" "{v}" """
        elif col_type == 'int':
            v = int(v)
        pred = f"{c} {o} {v}"
    else : assert False
    return pred


def filtered_indices(table,predicates,table_name,dtype_dict) :
    df = table
    if len(predicates) == 0:
        return df
    pred_list = list()
    for c,o,v in predicates :
        col_type = dtype_dict[f"{table_name}.{c}"]
        pred = get_query_pred(col_type,c,o,v)
        pred_list.append(pred)
    pred = ' and '.join(pred_list)
    # print(pred)
    df = df.query(pred,engine='python').index.tolist()
    return df

def load_tables(tables,dtype_dict,data_dir,**kwargs) :
    table_dict = dict()
    for table in tables :
        if table in table_dtype.keys():
            dtype_dict = table_dtype[table]
        else:
            dtype_dict = dict()

        df = pd.read_csv(os.path.join(data_dir, f"{table}.csv"),dtype=dtype_dict,
                               low_memory=False,keep_default_na=False,na_values=[''],
                               **kwargs)
        # print(f'Data type : {table} - {dtype_dict}\n{df.dtypes}')
        for col in df.columns:
            if df[col].dtype == object :
                df[col] = df[col].str.strip()
        table_dict[table] = df
    return table_dict

def parse(query_file_path, sep):
    predicates = []
    tables = []
    with open(query_file_path, 'rU') as f:
        lines = list(list(rec) for rec in csv.reader(f, delimiter=sep))
        for line in lines:
            
            row = list(list(rec) for rec in csv.reader(line, delimiter=","))
            # row = line
            tables.append(row[0])
            if len(row[2]) > 0:
                predicates.append(row[2])
            else:
                predicates.append([''])
        predicates = [list(chunks(d, 3)) for d in predicates]
    return tables, predicates


def write_bitmap(table_to_df, dtype_dict, tables, predicates, output_path):
    outfile=open(output_path, 'wb')
    num_queries = len(tables)
    bitmap_evaluation_time_total = 0.0
    err_list = []
    times = []

    for i, query_tables in enumerate(tables):
        if i % 1000 == 0: print(f'{i}/{num_queries}')
        # predicates => {table_alias : [(column,op,operand)]
        query_predicates=predicates[i]
        table_to_predicates = dict()
        query_time = 0.0

        for predicate in query_predicates:
            if len(predicate) == 3:
                # Proper predicate
                table = predicate[0].split(".")[0]
                column = predicate[0].split(".")[1]
                operator = predicate[1]
                val = predicate[2]
                val = val.strip()

                if table not in table_to_predicates:
                    table_to_predicates[table] = []
                table_to_predicates[table].append((column, operator, val))
        num_tables=len(query_tables)

        outfile.write(num_tables.to_bytes(4,"little"))
        try:
            for query_table in query_tables:
                
                table = query_table.split(" ")[0]
                alias = query_table.split(" ")[1]

                t_start = time.time()    
                if alias  in table_to_predicates :
                    filtered = filtered_indices(table_to_df[table],  table_to_predicates[alias], table, dtype_dict)
                    bitmap = np.zeros(NUM_MATERIALIZED_SAMPLES, dtype=bool)
                    bitmap[filtered] = 1
                else:
                    bitmap = np.ones(NUM_MATERIALIZED_SAMPLES, dtype=bool)
                eval_time = time.time() - t_start
                query_time += eval_time
                bitmap_evaluation_time_total += eval_time
                # print(bitmap)
                outfile.write(np.packbits(bitmap).tobytes())
        except KeyboardInterrupt:
            sys.exit()
        except:
            err_list.append(i)
        times.append(query_time)
    print(f'bitmap evaluation time total = {bitmap_evaluation_time_total} sec')
    print(f'bitmap evaluation time avg = {bitmap_evaluation_time_total / num_queries} sec')
    # print(err_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input query file", type = str)
    parser.add_argument("--output", help="path to output bitmap file", type = str, default =None)
    
    parser.add_argument("--sample", help="path to csv files of samples", type = str)
    parser.add_argument("--dbname", help="imdb, tpcds, ... (default: imdb)", type = str, default ="imdb")
    parser.add_argument("--no_alias", help="do not use table alias in query", action="store_true")
    
    args = parser.parse_args()
    sep="#"
    dataset = args.dbname
    if dataset == 'imdb' :
        table_list = JOB_TABLES
        alias_dict = JOB_ALIAS_DICT
        dtype_dict = imdbDtypeDict
        sep = '#'
    elif dataset == 'tpcds' :
        table_list = TPCDS_TABLES
        alias_dict = TPCDS_ALIAS_DICT
        dtype_dict = tpcdsDtypeDict
        sep = '|'
    elif dataset =='syn-multi':
        table_list = SYN_M_TABLES
        alias_dict = SYN_M_ALIAS_DICT
        dtype_dict = SYN_M_DtypeDict
        sep = '#'
    elif dataset =='syn-single':
        table_list = SYN_S_TABLES
        alias_dict = SYN_S_ALIAS_DICT
        dtype_dict = SYM_S_DtypeDict
        sep = '#'

    table_to_df = load_tables(table_list,dtype_dict,data_dir=args.sample)
    
    assert table_list is not None
    
    input_path = args.input
    tables, predicates = parse(input_path, sep)

    rev_alias_dict = {v:k for k,v in alias_dict.items()}

    if args.no_alias:
        print("query has no alias")
        tables = convert_tables(tables, rev_alias_dict)
        predicates = convert_preds(predicates, rev_alias_dict)

    if args.output == None:
        # output_path = input_path.split(".")[-2] + ".bitmaps"
        output_path = input_path.replace(".csv", ".bitmaps")
    write_bitmap(table_to_df, dtype_dict, tables, predicates, output_path)

if __name__ == "__main__":
    main()
