import csv
import torch
from pathlib import Path
from torch.utils.data import dataset

from mscn.util_common import *


IMDB_STRING_COLUMNS = {
    "t.title", "t.imdb_index", "t.phonetic_code", "t.series_years", "t.md5sum",
    "mc.note", 
    "ci.note",
    "mi.info", "mi.note",
    "mi_idx.info", "mi_idx.note",
    #mk, ml
    "cct.kind",
    "cn.country_code", "cn.name", "cn.name_pcode_nf", "cn.name_pcode_sf", "cn.md5sum",
    "ch_n.name", "ch_n.imdb_index", "ch_n.name_pcode_nf", "ch_n.surname_pcode", "ch_n.md5sum",
    #cc
    "ct.kind",
    "it.info", 
    "k.keyword", "k.phonetic_code",
    "kt.kind",
    "lt.link",
    "an.name", "an.imdb_index", "an.name_pcode_cf", "an.name_pcode_nf", "an.surname_pcode", "an.md5sum",
    "at.title", "at.imdb_index", "at.phonetic_code", "at.note", "at.md5sum",
    "n.name", "n.imdb_index", "n.gender", "n.name_pcode_cf", "n.name_pcode_nf", "n.surname_pcode", "n.md5sum",
    "pi.info", "pi.note",
    "rt.role"
}

TPCDS_STRING_COLUMNS = {
    'web.web_street_type', 's.s_state', 'web.web_country', 'sm.sm_contract', 'd.d_current_day', 'd.d_current_week', 'w.w_street_type', 'w.w_city', 'cc.cc_name', 'i.i_item_desc', 'w.w_county', 'w.w_zip', 'w.w_warehouse_id', 'cp.cp_type', 'cd.cd_education_status', 'd.d_day_name', 'ca.ca_city', 'cc.cc_state', 'cc.cc_street_type', 'c.c_birth_country', 'web.web_company_name', 's.s_store_name', 'ca.ca_country', 'cc.cc_street_name', 'w.w_street_number', 'cd.cd_credit_rating', 'ca.ca_street_type', 'web.web_street_number', 's.s_country', 'cc.cc_suite_number', 'c.c_last_name', 'cc.cc_hours', 'p.p_channel_event', 'cc.cc_zip', 'ca.ca_suite_number', 'ca.ca_location_type', 'cp.cp_description', 'web.web_site_id', 's.s_street_type', 'web.web_state', 'web.web_market_manager', 'cc.cc_country', 's.s_market_manager', 'wp.wp_url', 'hd.hd_buy_potential', 's.s_zip', 'ca.ca_county', 's.s_division_name', 'sm.sm_carrier', 'ca.ca_street_name', 'd.d_current_quarter', 'p.p_channel_dmail', 'cp.cp_department', 'c.c_email_address', 'web.web_county', 'cc.cc_street_number', 't.t_meal_time', 'cd.cd_marital_status', 's.s_street_number', 'p.p_promo_id', 'r.r_reason_id', 'web.web_mkt_class', 'web.web_zip', 'ca.ca_zip', 'cc.cc_market_manager', 'ca.ca_state', 'web.web_street_name', 't.t_sub_shift', 'web.web_name', 'p.p_channel_catalog', 't.t_shift', 's.s_hours', 'cd.cd_gender', 's.s_manager', 'p.p_discount_active', 'd.d_weekend', 's.s_suite_number', 'i.i_class', 'cc.cc_mkt_desc', 'p.p_channel_demo', 'c.c_login', 's.s_store_id', 'web.web_mkt_desc', 'd.d_following_holiday', 'i.i_color', 's.s_city', 's.s_market_desc', 'c.c_preferred_cust_flag', 'cc.cc_company_name', 'web.web_class', 'd.d_current_year', 'p.p_channel_press', 'i.i_formulation', 'cc.cc_manager', 'sm.sm_code', 'p.p_channel_radio', 'cc.cc_mkt_class', 'cp.cp_catalog_page_id', 'sm.sm_type', 'cc.cc_call_center_id', 'w.w_warehouse_name', 'd.d_current_month', 'wp.wp_type', 'w.w_street_name', 'd.d_holiday', 't.t_time_id', 'i.i_category', 'w.w_state', 't.t_am_pm', 's.s_geography_class', 'p.p_purpose', 'web.web_city', 'i.i_manufact', 'i.i_container', 'c.c_customer_id', 'p.p_channel_details', 'w.w_suite_number', 'c.c_salutation', 'web.web_suite_number', 'i.i_units', 'web.web_manager', 's.s_county', 'cc.cc_city', 'sm.sm_ship_mode_id', 'wp.wp_web_page_id', 'i.i_size', 's.s_company_name', 'i.i_brand', 'c.c_first_name', 'p.p_promo_name', 'r.r_reason_desc', 'cc.cc_county', 'cc.cc_division_name', 'p.p_channel_email', 'cc.cc_class', 'i.i_product_name', 'p.p_channel_tv', 's.s_street_name', 'wp.wp_autogen_flag', 'i.i_item_id', 'd.d_date_id', 'ca.ca_address_id', 'w.w_country', 'd.d_quarter_name', 'ca.ca_street_number'
}

IMDB_ALIAS_DICT = {'name': 'n', 'movie_companies': 'mc', 'aka_name': 'an', 'movie_info': 'mi', 'movie_keyword': 'mk', 'person_info': 'pi', 'comp_cast_type': 'cct', 'complete_cast': 'cc', 'char_name': 'ch_n', 'movie_link': 'ml', 'company_type': 'ct', 'cast_info': 'ci', 'info_type': 'it', 'company_name': 'cn', 'aka_title': 'at', 'kind_type': 'kt', 'role_type': 'rt', 'movie_info_idx': 'mi_idx', 'keyword': 'k', 'link_type': 'lt', 'title': 't'}
TPCDS_ALIAS_DICT= {'store_sales': 'ss', 'store_returns': 'sr', 'catalog_sales': 'cs', 'catalog_returns': 'cr', 'web_sales': 'ws', 'web_returns': 'wr', 'inventory': 'inv', 'store': 's', 'call_center': 'cc', 'catalog_page': 'cp', 'web_site': 'web', 'web_page': 'wp', 'warehouse': 'w', 'customer': 'c', 'customer_address': 'ca', 'customer_demographics': 'cd', 'date_dim': 'd', 'household_demographics': 'hd', 'item': 'i', 'income_band': 'ib', 'promotion': 'p', 'reason': 'r', 'ship_mode': 'sm', 'time_dim': 't'}
SYN_MULTI_ALIAS_DICT = {'table0': 't0', 'table1': 't1', 'table2': 't2', 'table3': 't3', 'table4': 't4', 'table5': 't5', 'table6': 't6', 'table7': 't7', 'table8': 't8', 'table9': 't9'}
SYN_SINGLE_ALIAS_DICT = {'table0': 't0'}

def prepare_loading(query_file_path, dbname):
    if ("job" in dbname) or ("imdb" in dbname):
        sep = "#"
        minmax_file_path = str(Path(__file__).parent.parent.absolute()) + f'/minmax/imdb.csv'
        string_columns = IMDB_STRING_COLUMNS
        alias_dict = IMDB_ALIAS_DICT
    elif "tpcds" in dbname:
        sep = "|"
        minmax_file_path = str(Path(__file__).parent.parent.absolute()) + f'/minmax/tpcds.csv'
        string_columns = TPCDS_STRING_COLUMNS
        alias_dict = TPCDS_ALIAS_DICT
    elif "syn-multi" in dbname:
        sep = "#"
        syn_id = dbname[-2:]
        minmax_file_path = str(Path(__file__).parent.parent.absolute()) + f'/minmax/synthetic/multi/minmax_{syn_id}.csv'
        string_columns = set()
        alias_dict = SYN_MULTI_ALIAS_DICT
    elif "syn-single" in dbname:
        sep = "#"
        syn_id = dbname[-2:]
        minmax_file_path = str(Path(__file__).parent.parent.absolute()) + f'/minmax/synthetic/single/minmax_{syn_id}.csv'
        string_columns = set()
        alias_dict = SYN_SINGLE_ALIAS_DICT
    else:
        print("Unsupported database!!")
        raise
    
    # sample_file_path = query_file_path.split(".")[0] + ".bitmaps"
    sample_file_path = query_file_path.replace(".csv", ".bitmaps")

    word_vectors_path = str(Path(__file__).parent.parent.parent.absolute()) + f'/wordvectors/{dbname}/wordvectors_updated.kv'

    return sep, string_columns, minmax_file_path, sample_file_path, word_vectors_path, alias_dict
            

def load_data_from_path(query_file_path, sample_file_path, sep, no_alias = False, alias_dict = dict()):
    joins = []
    predicates = []
    tables = []
    samples = []
    label = []

    with open(query_file_path, 'rU') as f:
        lines = list(list(rec) for rec in csv.reader(f, delimiter=sep))
        
        for line in lines:
            row = list(list(rec) for rec in csv.reader(line, delimiter=","))
            if len(row) < 4:
                print(line)
                raise
            # row = line
            tables.append(row[0])
            if len(row[1]) > 0:
                joins.append(row[1])
            else:
                joins.append([''])
            if len(row[2]) > 0:
                predicates.append(row[2])
            else:
                predicates.append([''])
            if int(row[3][0]) < 1:
                row[3][0] = "1"
                # print("Queries must have non-zero cardinalities")
                # exit(1)
            label.append(row[3][0])
    print("Loaded queries")
    print(len(tables))

    # Load bitmaps
    num_bytes_per_bitmap = int((NUM_MATERIALIZED_SAMPLES + 7) >> 3)
    with open(sample_file_path, 'rb') as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")
    print(len(samples))

    # Split predicates
    predicates = [list(chunks(d, 3)) for d in predicates]
    if no_alias:
        tables = convert_tables(tables, alias_dict)
        joins = convert_joins(joins, alias_dict)
        predicates = convert_preds(predicates, alias_dict)

    return joins, predicates, tables, samples, label

