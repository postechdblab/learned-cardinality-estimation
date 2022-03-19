import logging
import pickle
from time import perf_counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

table_dtype ={'aka_name': {'name': object,   'imdb_index': object,   'name_pcode_cf': object,   'name_pcode_nf': object,   'surname_pcode': object,   'md5sum': object},  'aka_title': {'title': object,   'imdb_index': object,   'phonetic_code': object,   'note': object,   'md5sum': object},  'cast_info': {'note': object},  'char_name': {'name': object,   'imdb_index': object,   'name_pcode_nf': object,   'surname_pcode': object,   'md5sum': object},  'comp_cast_type': {'kind': object},  'company_name': {'name': object,   'country_code': object,   'name_pcode_nf': object,   'name_pcode_sf': object,   'md5sum': object},  'company_type': {'kind': object},  'info_type': {'info': object},  'keyword': {'keyword': object, 'phonetic_code': object},  'kind_type': {'kind': object},  'link_type': {'link': object},  'movie_companies': {'note': object},  'movie_info_idx': {'info': object, 'note': object},  'name': {'name': object,   'imdb_index': object,   'gender': object,   'name_pcode_cf': object,   'name_pcode_nf': object,   'surname_pcode': object,   'md5sum': object},  'role_type': {'role': object},  'title': {'title': object,   'imdb_index': object,   'phonetic_code': object,   'series_years': object,   'md5sum': object},  'movie_info': {'info': object, 'note': object},  'person_info': {'info': object, 'note': object},   'customer_address': {'ca_address_id': object,   'ca_street_number': object,   'ca_street_name': object,   'ca_street_type': object,   'ca_suite_number': object,   'ca_city': object,   'ca_county': object,   'ca_state': object,   'ca_zip': object,   'ca_country': object,   'ca_location_type': object},  'customer_demographics': {'cd_gender': object,   'cd_marital_status': object,   'cd_education_status': object,   'cd_credit_rating': object},  'date_dim': {'d_date_id': object,   'd_day_name': object,   'd_quarter_name': object,   'd_holiday': object,   'd_weekend': object,   'd_following_holiday': object,   'd_current_day': object,   'd_current_week': object,   'd_current_month': object,   'd_current_quarter': object,   'd_current_year': object,   'd_date': object},  'warehouse': {'w_warehouse_id': object,   'w_warehouse_name': object,   'w_street_number': object,   'w_street_name': object,   'w_street_type': object,   'w_suite_number': object,   'w_city': object,   'w_county': object,   'w_state': object,   'w_zip': object,   'w_country': object},  'ship_mode': {'sm_ship_mode_id': object,   'sm_type': object,   'sm_code': object,   'sm_carrier': object,   'sm_contract': object},  'time_dim': {'t_time_id': object,   't_am_pm': object,   't_shift': object,   't_sub_shift': object,   't_meal_time': object},  'reason': {'r_reason_id': object, 'r_reason_desc': object},  'item': {'i_item_id': object,   'i_item_desc': object,   'i_brand': object,   'i_class': object,   'i_category': object,   'i_manufact': object,   'i_size': object,   'i_formulation': object,   'i_color': object,   'i_units': object,   'i_container': object,   'i_product_name': object,   'i_rec_start_date': object,   'i_rec_end_date': object},  'store': {'s_store_id': object,   's_store_name': object,   's_hours': object,   's_manager': object,   's_geography_class': object,   's_market_desc': object,   's_market_manager': object,   's_division_name': object,   's_company_name': object,   's_street_number': object,   's_street_name': object,   's_street_type': object,   's_suite_number': object,   's_city': object,   's_county': object,   's_state': object,   's_zip': object,   's_country': object,   's_rec_start_date': object,   's_rec_end_date': object},  'call_center': {'cc_call_center_id': object,   'cc_name': object,   'cc_class': object,   'cc_hours': object,   'cc_manager': object,   'cc_mkt_class': object,   'cc_mkt_desc': object,   'cc_market_manager': object,   'cc_division_name': object,   'cc_company_name': object,   'cc_street_number': object,   'cc_street_name': object,   'cc_street_type': object,   'cc_suite_number': object,   'cc_city': object,   'cc_county': object,   'cc_state': object,   'cc_zip': object,   'cc_country': object,   'cc_rec_start_date': object,   'cc_rec_end_date': object},  'customer': {'c_customer_id': object,   'c_salutation': object,   'c_first_name': object,   'c_last_name': object,   'c_preferred_cust_flag': object,   'c_birth_country': object,   'c_login': object,   'c_email_address': object},  'web_site': {'web_site_id': object,   'web_name': object,   'web_class': object,   'web_manager': object,   'web_mkt_class': object,   'web_mkt_desc': object,   'web_market_manager': object,   'web_company_name': object,   'web_street_number': object,   'web_street_name': object,   'web_street_type': object,   'web_suite_number': object,   'web_city': object,   'web_county': object,   'web_state': object,   'web_zip': object,   'web_country': object,   'web_rec_start_date': object,   'web_rec_end_date': object},  'household_demographics': {'hd_buy_potential': object},  'web_page': {'wp_web_page_id': object,   'wp_autogen_flag': object,   'wp_url': object,   'wp_type': object,   'wp_rec_start_date': object,   'wp_rec_end_date': object},  'promotion': {'p_promo_id': object,   'p_promo_name': object,   'p_channel_dmail': object,   'p_channel_email': object,   'p_channel_catalog': object,   'p_channel_tv': object,   'p_channel_radio': object,   'p_channel_press': object,   'p_channel_event': object,   'p_channel_demo': object,   'p_channel_details': object,   'p_purpose': object,   'p_discount_active': object},  'catalog_page': {'cp_catalog_page_id': object,   'cp_department': object,   'cp_description': object,   'cp_type': object}}

def read_table_csv(table_obj, csv_seperator=','):
    """
    Reads csv from path, renames columns and drops unnecessary columns
    """
    table =table_obj.table_name

    if table in table_dtype.keys():
        dtype_dict = table_dtype[table]
    else:
        dtype_dict = dict()
    
    df_rows = pd.read_csv(table_obj.csv_file_location, sep=csv_seperator, dtype=dtype_dict, low_memory=False, keep_default_na=False, na_values=[''])
    for col in df_rows.columns:
        if df_rows[col].dtype == object :
            df_rows[col] = df_rows[col].str.strip()
    
    df_rows.columns = [table_obj.table_name + '.' + attr for attr in table_obj.attributes]

    for attribute in table_obj.irrelevant_attributes:
        df_rows = df_rows.drop(table_obj.table_name + '.' + attribute, axis=1)

    return df_rows.apply(pd.to_numeric, errors="ignore")
    # return df_rows.convert_objects()

def find_relationships(schema_graph, table, incoming=True):
    relationships = []

    for relationship_obj in schema_graph.relationships:

        if relationship_obj.end == table and incoming:
            relationships.append(relationship_obj)
        if relationship_obj.start == table and not incoming:
            relationships.append(relationship_obj)

    return relationships


def prepare_single_table(schema_graph, table, path, max_distinct_vals=10000, csv_seperator=',',
                         max_table_data=20000000):
    """
    Reads table csv. Adds multiplier fields, missing value imputation, dict for categorical data. Adds null tuple tables.

    :param schema_graph:
    :param table:
    :return:
    """
    table_meta_data = dict()
    table_obj = schema_graph.table_dictionary[table]
    table_data = read_table_csv(table_obj, csv_seperator=csv_seperator)
    table_sample_rate = table_obj.sample_rate

    relevant_attributes = [x for x in table_obj.attributes if x not in table_obj.irrelevant_attributes]

    table_meta_data['hdf_path'] = path
    table_meta_data['incoming_relationship_means'] = {}
    
    join_key_set = set()
    for join_clause in schema_graph.relationship_dictionary.keys():
        l_key,r_key = join_clause.split(' = ')
        join_key_set.add(l_key)
        join_key_set.add(r_key)
    logger.info(f"Join key set {join_key_set}")
    

    # manage functional dependencies
    logger.info(f"Managing functional dependencies for table {table}")
    table_meta_data['fd_dict'] = dict()
    cols_to_be_dropped = []
    for attribute_wo_table in table_obj.attributes:
        attribute = table + '.' + attribute_wo_table
        fd_children = table_obj.children_fd_attributes(attribute)
        if len(fd_children) > 0:
            for child in fd_children:
                logger.info(f"Managing functional dependencies for {child}->{attribute}")
                distinct_tuples = table_data.drop_duplicates([attribute, child])[[attribute, child]].values
                reverse_dict = {}
                for attribute_value, child_value in distinct_tuples:
                    if reverse_dict.get(attribute_value) is None:
                        reverse_dict[attribute_value] = []
                    reverse_dict[attribute_value].append(child_value)
                if table_meta_data['fd_dict'].get(attribute) is None:
                    table_meta_data['fd_dict'][attribute] = dict()
                table_meta_data['fd_dict'][attribute][child] = reverse_dict
            # remove from dataframe and relevant attributes
            cols_to_be_dropped.append(attribute)
            relevant_attributes.remove(attribute_wo_table)
    table_data.drop(columns=cols_to_be_dropped, inplace=True)

    # add multiplier fields
    logger.info("Preparing multipliers for table {}".format(table))
    incoming_relationships = find_relationships(schema_graph, table, incoming=True)

    for relationship_obj in incoming_relationships:
        logger.info("Preparing multiplier {} for table {}".format(relationship_obj.identifier, table))

        neighbor_table = relationship_obj.start
        neighbor_table_obj = schema_graph.table_dictionary[neighbor_table]
        neighbor_sample_rate = neighbor_table_obj.sample_rate

        left_attribute = table + '.' + relationship_obj.end_attr
        right_attribute = neighbor_table + '.' + relationship_obj.start_attr

        neighbor_table_data = read_table_csv(neighbor_table_obj, csv_seperator=csv_seperator).set_index(right_attribute,
                                                                                                        drop=False)
        table_data = table_data.set_index(left_attribute, drop=False)

        assert len(table_obj.primary_key) == 1, \
            "Currently, only single primary keys are supported for table with incoming edges"
        table_primary_key = table + '.' + table_obj.primary_key[0]
        assert table_primary_key == left_attribute, "Currently, only references to primary key are supported"

        # fix for new pandas version
        table_data.index.name = None
        neighbor_table_data.index.name = None
        muls = table_data.join(neighbor_table_data, how='left')[[table_primary_key, right_attribute]] \
            .groupby([table_primary_key]).count()

        mu_nn_col_name = relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name_nn
        mu_col_name = relationship_obj.end + '.' + relationship_obj.multiplier_attribute_name

        muls.columns = [mu_col_name]
        # if we just have a sample of the neighbor table we assume larger multipliers
        muls[mu_col_name] = muls[mu_col_name] * 1 / neighbor_sample_rate
        muls[mu_nn_col_name] = muls[mu_col_name].replace(to_replace=0, value=1)

        table_data = table_data.join(muls)

        relevant_attributes.append(relationship_obj.multiplier_attribute_name)
        relevant_attributes.append(relationship_obj.multiplier_attribute_name_nn)

        table_meta_data['incoming_relationship_means'][relationship_obj.identifier] = table_data[mu_nn_col_name].mean()

    # save if there are entities without FK reference (e.g. orders without customers)
    outgoing_relationships = find_relationships(schema_graph, table, incoming=False)
    for relationship_obj in outgoing_relationships:
        fk_attribute_name = table + '.' + relationship_obj.start_attr

        table_meta_data[relationship_obj.identifier] = {
            'fk_attribute_name': fk_attribute_name,
            'length': table_data[fk_attribute_name].isna().sum() * 1 / table_sample_rate,
            'path': None
        }

    # null value imputation and categorical value replacement
    logger.info("Preparing categorical values and null values for table {}".format(table))
    table_meta_data['categorical_columns_dict'] = {}
    table_meta_data['null_values_column'] = []
    del_cat_attributes = []

    for rel_attribute in relevant_attributes:

        attribute = table + '.' + rel_attribute

        # categorical value
        if table_data.dtypes[attribute] == object:

            logger.debug("\t\tPreparing categorical values for column {}".format(rel_attribute))

            distinct_vals = table_data[attribute].unique()

            # For fair comparison, disable this, this also prevent running JOB-M queries with JOB-Union schema
            if len(distinct_vals) > max_distinct_vals:
                del_cat_attributes.append(rel_attribute)
                logger.info("Ignoring column {} for table {} because "
                            "there are too many categorical values {}".format(rel_attribute, table, len(distinct_vals)))
            # all values nan does not provide any information
            elif not table_data[attribute].notna().any():
                del_cat_attributes.append(rel_attribute)
                logger.info("Ignoring column {} for table {} because all values are nan".format(rel_attribute, table))
            else:
                if not table_data[attribute].isna().any():
                    val_dict = dict(zip(distinct_vals, range(1, len(distinct_vals) + 1)))
                    val_dict[np.nan] = 0
                else:
                    val_dict = dict(zip(distinct_vals, range(1, len(distinct_vals) + 1)))
                    val_dict[np.nan] = 0
                table_meta_data['categorical_columns_dict'][attribute] = val_dict

                table_data[attribute] = table_data[attribute].map(val_dict.get)
                # because we are paranoid
                table_data[attribute] = table_data[attribute].fillna(0)
                # apparently slow
                # table_data[attribute] = table_data[attribute].replace(val_dict)
                table_meta_data['null_values_column'].append(val_dict[np.nan])

        # numerical value
        else:

            logger.debug("\t\tPreparing numerical values for column {}".format(rel_attribute))

            # all nan values
            if not table_data[attribute].notna().any():
                del_cat_attributes.append(rel_attribute)
                logger.info("Ignoring column {} for table {} because all values are nan".format(rel_attribute, table))
            else:
                contains_nan = table_data[attribute].isna().any()

                # not the best solution but works
                if attribute in join_key_set:
                    assert not (table_data[attribute]==-1).any()
                    unique_null_val = -1
                    table_data[attribute] = table_data[attribute].fillna(unique_null_val).astype(np.int64)
                    table_meta_data['null_values_column'].append(unique_null_val)
                else:
                    unique_null_val = table_data[attribute].mean() + 0.0001
                    assert not (table_data[attribute] == unique_null_val).any()
                    table_data[attribute] = table_data[attribute].fillna(unique_null_val)
                    table_meta_data['null_values_column'].append(unique_null_val)

                if contains_nan:
                    assert (table_data[attribute] == unique_null_val).any(), "Null value cannot be found"

    # remove categorical columns with too many entries from relevant tables and dataframe
    relevant_attributes = [x for x in relevant_attributes if x not in del_cat_attributes]
    logger.info("Relevant attributes for table {} are {}".format(table, relevant_attributes))
    logger.info("NULL values for table {} are {}".format(table, table_meta_data['null_values_column']))
    del_cat_attributes = [table + '.' + rel_attribute for rel_attribute in del_cat_attributes]
    table_data = table_data.drop(columns=del_cat_attributes)

    assert len(relevant_attributes) == len(table_meta_data['null_values_column']), \
        "Length of NULL values does not match"
    table_meta_data['relevant_attributes'] = relevant_attributes
    table_meta_data['relevant_attributes_full'] = [table + '.' + attr for attr in relevant_attributes]
    table_meta_data['length'] = len(table_data) * 1 / table_sample_rate

    assert not table_data.isna().any().any(), "Still contains null values"

    # save modified table
    if len(table_data) < max_table_data:
        table_data.to_hdf(path, key='df', format='table')
    else:
        table_data.sample(max_table_data).to_hdf(path, key='df', format='table')

    # add table parts without join partners
    logger.info("Adding table parts without join partners for table {}".format(table))
    for relationship_obj in incoming_relationships:
        logger.info("Adding table parts without join partners "
                    "for table {} and relationship {}".format(table, relationship_obj.identifier))

        neighbor_table = relationship_obj.start
        neighbor_table_obj = schema_graph.table_dictionary[neighbor_table]
        neighbor_primary_key = neighbor_table + '.' + neighbor_table_obj.primary_key[0]

        left_attribute = table + '.' + relationship_obj.end_attr
        right_attribute = neighbor_table + '.' + relationship_obj.start_attr

        table_data = table_data.set_index(left_attribute, drop=False)
        neighbor_table_data = read_table_csv(neighbor_table_obj, csv_seperator=csv_seperator).set_index(right_attribute,
                                                                                                        drop=False)
        null_tuples = table_data.join(neighbor_table_data, how='left')
        null_tuples = null_tuples.loc[null_tuples[neighbor_primary_key].isna(),
                                      [table + '.' + attr for attr in relevant_attributes]]
        if len(null_tuples) > 0 and neighbor_table_obj.sample_rate < 1:
            logger.warning(f"For {relationship_obj.identifier} {len(null_tuples)} tuples without a join partner were "
                           f"found. This is potentially due to the sampling rate of {neighbor_table_obj.sample_rate}.")

        if len(null_tuples) > 0:
            null_tuple_path = path + relationship_obj.start + relationship_obj.start_attr + '.hdf'
            table_meta_data[relationship_obj.identifier] = {
                'length': len(null_tuples) * 1 / table_sample_rate,
                'path': null_tuple_path
            }
            null_tuples.to_hdf(null_tuple_path, key='df', format='table')

    return table_meta_data


def prepare_all_tables(schema_graph, path, csv_seperator=',', max_table_data=20000000):
    prep_start_t = perf_counter()
    meta_data = {}
    for table_obj in schema_graph.tables:
        table = table_obj.table_name
        logger.info("Preparing hdf file for table {}".format(table))
        meta_data[table] = prepare_single_table(schema_graph, table, path + '/' + table + '.hdf',
                                                csv_seperator=csv_seperator, max_table_data=max_table_data)

    with open(path + '/meta_data.pkl', 'wb') as f:
        pickle.dump(meta_data, f, pickle.HIGHEST_PROTOCOL)
    prep_end_t = perf_counter()

    with open(path + '/build_time_hdf.txt', "w") as text_file:
        text_file.write(str(round(prep_end_t-prep_start_t)))

    return meta_data
