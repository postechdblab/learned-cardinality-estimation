from gensim.models import Word2Vec
from gensim.models import KeyedVectors

import json
import math
import csv



IMDB_SMALL_INDEXES={'movie_id_movie_info', 'info_type_id_movie_info_idx', 'info_type_id_movie_info', 'kind_id_title', 'keyword_id_movie_keyword', 'company_id_movie_companies', 'title_pkey', 'movie_id_cast_info', 'company_type_id_movie_companies', 'movie_id_movie_info_idx', 'role_id_cast_info', 'movie_id_movie_companies', 'movie_id_movie_keyword'}
IMDB_MEDIUM_INDEXES={'kind_type_pkey', 'keyword_pkey', 'movie_id_movie_info_idx', 'keyword_id_movie_keyword', 'movie_id_movie_link', 'company_type_pkey', 'title_pkey', 'company_name_pkey', 'movie_id_movie_companies', 'movie_id_aka_title', 'movie_id_cast_info', 'company_type_id_movie_companies', 'movie_id_movie_keyword', 'link_type_pkey', 'link_type_id_movie_link', 'movie_id_movie_info', 'role_id_cast_info', 'info_type_id_movie_info_idx', 'kind_id_title', 'info_type_id_movie_info', 'company_id_movie_companies', 'movie_id_complete_cast', 'comp_cast_type_pkey', 'info_type_pkey'}
TPCDS_BENCH_INDEXES={'idx_store_s_gmt_offset', 'idx_web_sales_ws_web_page_sk', 'idx_catalog_sales_cs_order_number', 'idx_catalog_sales_cs_list_price', 'idx_customer_c_first_name', 'idx_warehouse_w_warehouse_sk', 'idx_income_band_ib_income_band_sk', 'idx_web_returns_wr_fee', 'idx_income_band_ib_upper_bound', 'idx_store_sales_ss_store_sk', 'idx_catalog_returns_cr_reversed_charge', 'idx_customer_c_first_sales_date_sk', 'idx_catalog_sales_cs_bill_hdemo_sk', 'idx_catalog_sales_cs_bill_customer_sk', 'idx_item_i_category_id', 'idx_promotion_p_channel_event', 'idx_inventory_inv_item_sk', 'idx_catalog_sales_cs_ext_list_price', 'idx_ship_mode_sm_ship_mode_sk', 'idx_customer_c_birth_country', 'idx_customer_demographics_cd_gender', 'idx_web_site_web_name', 'idx_customer_address_ca_street_type', 'idx_catalog_returns_cr_refunded_addr_sk', 'idx_customer_c_first_shipto_date_sk', 'idx_inventory_inv_date_sk', 'idx_catalog_returns_cr_returning_customer_sk', 'idx_catalog_sales_cs_sales_price', 'idx_store_sales_ss_ext_tax', 'idx_web_sales_ws_sales_price', 'idx_customer_c_current_cdemo_sk', 'idx_store_returns_sr_customer_sk', 'idx_item_i_item_id', 'idx_catalog_sales_cs_bill_addr_sk', 'idx_web_sales_ws_sold_date_sk', 'idx_web_sales_ws_ext_discount_amt', 'idx_household_demographics_hd_vehicle_count', 'idx_ship_mode_sm_type', 'idx_customer_demographics_cd_purchase_estimate', 'idx_catalog_returns_cr_returned_date_sk', 'idx_web_sales_ws_item_sk', 'idx_web_sales_ws_ext_list_price', 'idx_time_dim_t_minute', 'idx_web_sales_ws_net_profit', 'idx_date_dim_d_year', 'idx_time_dim_t_time_sk', 'idx_catalog_sales_cs_sold_time_sk', 'idx_catalog_returns_cr_store_credit', 'idx_web_sales_ws_ship_addr_sk', 'idx_web_page_wp_web_page_sk', 'idx_web_sales_ws_list_price', 'idx_date_dim_d_day_name', 'idx_warehouse_w_warehouse_sq_ft', 'idx_customer_address_ca_location_type', 'idx_date_dim_d_date', 'idx_call_center_cc_call_center_id', 'idx_store_sales_ss_net_paid', 'idx_catalog_returns_cr_order_number', 'idx_store_sales_ss_addr_sk', 'idx_web_sales_ws_promo_sk', 'idx_customer_c_customer_sk', 'idx_customer_demographics_cd_dep_count', 'web_sales_pkey', 'idx_catalog_returns_cr_refunded_cash', 'idx_item_i_brand', 'idx_item_i_manufact', 'web_returns_pkey', 'idx_store_s_street_name', 'idx_store_sales_ss_ext_wholesale_cost', 'idx_store_sales_ss_sales_price', 'idx_customer_c_current_hdemo_sk', 'idx_web_sales_ws_web_site_sk', 'idx_store_returns_sr_net_loss', 'inventory_pkey', 'idx_customer_c_birth_year', 'idx_web_returns_wr_returning_addr_sk', 'idx_customer_c_birth_day', 'idx_store_sales_ss_hdemo_sk', 'idx_store_s_county', 'idx_catalog_sales_cs_ext_sales_price', 'idx_web_returns_wr_net_loss', 'idx_customer_c_current_addr_sk', 'idx_customer_c_last_review_date_sk', 'idx_store_sales_ss_customer_sk', 'idx_customer_c_preferred_cust_flag', 'idx_web_sales_ws_order_number', 'idx_web_sales_ws_ship_cdemo_sk', 'idx_store_sales_ss_net_profit', 'idx_store_sales_ss_quantity', 'idx_store_s_market_id', 'idx_catalog_sales_cs_catalog_page_sk', 'idx_store_s_number_employees', 'idx_catalog_sales_cs_coupon_amt', 'idx_store_s_street_number', 'idx_store_s_store_sk', 'idx_time_dim_t_hour', 'idx_catalog_returns_cr_call_center_sk', 'idx_web_sales_ws_wholesale_cost', 'idx_web_returns_wr_returning_cdemo_sk', 'idx_household_demographics_hd_demo_sk', 'idx_catalog_sales_cs_ship_customer_sk', 'idx_customer_address_ca_county', 'idx_web_sales_ws_sold_time_sk', 'idx_call_center_cc_manager', 'idx_time_dim_t_time', 'idx_catalog_sales_cs_ship_cdemo_sk', 'idx_customer_address_ca_street_number', 'idx_catalog_sales_cs_sold_date_sk', 'idx_store_sales_ss_wholesale_cost', 'idx_inventory_inv_quantity_on_hand', 'idx_catalog_returns_cr_returning_addr_sk', 'idx_catalog_sales_cs_ext_wholesale_cost', 'idx_customer_address_ca_country', 'idx_warehouse_w_country', 'idx_web_sales_ws_quantity', 'idx_item_i_color', 'catalog_sales_pkey', 'idx_web_sales_ws_ext_sales_price', 'idx_household_demographics_hd_income_band_sk', 'idx_catalog_sales_cs_ship_mode_sk', 'idx_store_s_company_id', 'idx_customer_address_ca_state', 'idx_customer_address_ca_city', 'idx_catalog_sales_cs_item_sk', 'idx_web_sales_ws_ship_date_sk', 'idx_catalog_sales_cs_ext_discount_amt', 'idx_customer_address_ca_address_sk', 'idx_web_sales_ws_ship_mode_sk', 'idx_store_returns_sr_return_amt', 'idx_store_s_company_name', 'idx_store_s_suite_number', 'idx_web_returns_wr_returning_customer_sk', 'idx_web_returns_wr_refunded_cdemo_sk', 'idx_web_returns_wr_return_quantity', 'idx_store_s_street_type', 'idx_household_demographics_hd_dep_count', 'idx_store_s_store_name', 'idx_store_returns_sr_item_sk', 'idx_web_sales_ws_bill_customer_sk', 'idx_item_i_current_price', 'idx_promotion_p_channel_tv', 'idx_web_sales_ws_ship_customer_sk', 'idx_customer_demographics_cd_credit_rating', 'idx_item_i_class', 'idx_promotion_p_promo_sk', 'idx_time_dim_t_meal_time', 'idx_date_dim_d_quarter_name', 'idx_store_sales_ss_sold_time_sk', 'idx_warehouse_w_county', 'idx_catalog_sales_cs_ext_ship_cost', 'idx_store_sales_ss_ext_sales_price', 'idx_catalog_sales_cs_bill_cdemo_sk', 'idx_web_sales_ws_ship_hdemo_sk', 'idx_income_band_ib_lower_bound', 'store_returns_pkey', 'idx_web_sales_ws_net_paid', 'idx_warehouse_w_warehouse_name', 'idx_store_sales_ss_ext_list_price', 'idx_store_s_state', 'idx_catalog_returns_cr_net_loss', 'idx_catalog_sales_cs_warehouse_sk', 'idx_web_site_web_site_id', 'idx_web_returns_wr_returned_date_sk', 'idx_catalog_page_cp_catalog_page_sk', 'idx_store_returns_sr_return_quantity', 'idx_web_sales_ws_ext_wholesale_cost', 'idx_date_dim_d_dow', 'idx_web_returns_wr_reason_sk', 'idx_store_sales_ss_ext_discount_amt', 'idx_item_i_size', 'idx_customer_address_ca_suite_number', 'idx_catalog_sales_cs_quantity', 'idx_store_sales_ss_ticket_number', 'idx_customer_c_last_name', 'idx_call_center_cc_county', 'idx_date_dim_d_dom', 'idx_store_returns_sr_returned_date_sk', 'idx_store_returns_sr_ticket_number', 'idx_web_sales_ws_bill_addr_sk', 'idx_item_i_item_desc', 'idx_web_site_web_company_name', 'idx_catalog_sales_cs_wholesale_cost', 'idx_web_returns_wr_order_number', 'idx_date_dim_d_qoy', 'idx_promotion_p_channel_dmail', 'idx_promotion_p_channel_email', 'idx_store_returns_sr_store_sk', 'idx_customer_c_email_address', 'idx_web_sales_ws_bill_cdemo_sk', 'idx_catalog_sales_cs_net_paid', 'idx_catalog_returns_cr_return_amount', 'idx_item_i_manager_id', 'idx_web_sales_ws_warehouse_sk', 'idx_date_dim_d_month_seq', 'idx_item_i_wholesale_cost', 'idx_call_center_cc_call_center_sk', 'idx_reason_r_reason_desc', 'idx_web_returns_wr_web_page_sk', 'idx_catalog_returns_cr_item_sk', 'idx_catalog_returns_cr_return_quantity', 'idx_customer_address_ca_street_name', 'idx_store_returns_sr_cdemo_sk', 'idx_store_sales_ss_coupon_amt', 'idx_store_s_zip', 'idx_date_dim_d_week_seq', 'idx_inventory_inv_warehouse_sk', 'store_sales_pkey', 'idx_store_sales_ss_list_price', 'idx_customer_c_login', 'idx_catalog_page_cp_catalog_page_id', 'idx_reason_r_reason_sk', 'idx_customer_c_customer_id', 'idx_customer_demographics_cd_dep_employed_count', 'idx_store_s_store_id', 'idx_catalog_returns_cr_catalog_page_sk', 'idx_customer_c_birth_month', 'idx_item_i_category', 'idx_item_i_manufact_id', 'idx_web_site_web_site_sk', 'idx_catalog_returns_cr_return_amt_inc_tax', 'idx_web_returns_wr_refunded_cash', 'idx_web_sales_ws_ext_ship_cost', 'idx_catalog_sales_cs_call_center_sk', 'idx_item_i_brand_id', 'idx_item_i_product_name', 'idx_item_i_units', 'idx_store_sales_ss_promo_sk', 'idx_catalog_sales_cs_net_profit', 'idx_web_returns_wr_item_sk', 'idx_warehouse_w_city', 'idx_catalog_sales_cs_ship_addr_sk', 'idx_household_demographics_hd_buy_potential', 'idx_store_sales_ss_item_sk', 'idx_store_sales_ss_cdemo_sk', 'idx_customer_demographics_cd_marital_status', 'idx_store_returns_sr_reason_sk', 'idx_item_i_item_sk', 'idx_web_page_wp_char_count', 'idx_store_s_city', 'idx_customer_address_ca_gmt_offset', 'idx_call_center_cc_name', 'idx_ship_mode_sm_carrier', 'idx_store_sales_ss_sold_date_sk', 'idx_warehouse_w_state', 'idx_customer_demographics_cd_education_status', 'idx_customer_address_ca_zip', 'idx_customer_demographics_cd_dep_college_count', 'idx_catalog_sales_cs_ship_date_sk', 'catalog_returns_pkey', 'idx_web_returns_wr_refunded_addr_sk', 'idx_customer_c_salutation', 'idx_customer_demographics_cd_demo_sk', 'idx_date_dim_d_moy', 'idx_date_dim_d_date_sk', 'idx_catalog_sales_cs_promo_sk', 'idx_item_i_class_id', 'idx_web_returns_wr_return_amt'}

IMDB_LARGE_INDEXES = ['aka_name_pkey', 'aka_title_pkey', 'cast_info_pkey', 'char_name_pkey',
               'comp_cast_type_pkey', 'company_name_pkey', 'company_type_pkey', 'complete_cast_pkey',
               'info_type_pkey', 'keyword_pkey', 'kind_type_pkey', 'link_type_pkey', 'movie_companies_pkey',
               'movie_info_idx_pkey', 'movie_keyword_pkey', 'movie_link_pkey', 'name_pkey', 'role_type_pkey',
               'title_pkey', 'movie_info_pkey', 'person_info_pkey', 'company_id_movie_companies',
               'company_type_id_movie_companies', 'info_type_id_movie_info_idx', 'info_type_id_movie_info',
               'info_type_id_person_info', 'keyword_id_movie_keyword', 'kind_id_aka_title', 'kind_id_title',
               'linked_movie_id_movie_link', 'link_type_id_movie_link', 'movie_id_aka_title', 'movie_id_cast_info',
               'movie_id_complete_cast', 'movie_id_movie_companies', 'movie_id_movie_info_idx',
               'movie_id_movie_keyword', 'movie_id_movie_link', 'movie_id_movie_info', 'person_id_aka_name',
               'person_id_cast_info', 'person_id_person_info', 'person_role_id_cast_info', 'role_id_cast_info']
TPCDS_LARGE_INDEXES = ['idx_warehouse_w_warehouse_sk', 'idx_warehouse_w_warehouse_id', 'idx_warehouse_w_warehouse_name', 'idx_warehouse_w_warehouse_sq_ft', 'idx_warehouse_w_street_number', 'idx_warehouse_w_street_name', 'idx_warehouse_w_street_type', 'idx_warehouse_w_suite_number', 'idx_warehouse_w_city', 'idx_warehouse_w_county', 'idx_warehouse_w_state', 'idx_warehouse_w_zip', 'idx_warehouse_w_country', 'idx_warehouse_w_gmt_offset', 'idx_store_returns_sr_returned_date_sk', 'idx_store_returns_sr_return_time_sk', 'idx_store_returns_sr_item_sk', 'idx_store_returns_sr_customer_sk', 'idx_store_returns_sr_cdemo_sk', 'idx_store_returns_sr_hdemo_sk', 'idx_store_returns_sr_addr_sk', 'idx_store_returns_sr_store_sk', 'idx_store_returns_sr_reason_sk', 'idx_store_returns_sr_ticket_number', 'idx_store_returns_sr_return_quantity', 'idx_store_returns_sr_return_amt', 'idx_store_returns_sr_return_tax', 'idx_store_returns_sr_return_amt_inc_tax', 'idx_store_returns_sr_fee', 'idx_store_returns_sr_return_ship_cost', 'idx_store_returns_sr_refunded_cash', 'idx_store_returns_sr_reversed_charge', 'idx_store_returns_sr_store_credit', 'idx_store_returns_sr_net_loss', 'idx_household_demographics_hd_demo_sk', 'idx_household_demographics_hd_income_band_sk', 'idx_household_demographics_hd_buy_potential', 'idx_household_demographics_hd_dep_count', 'idx_household_demographics_hd_vehicle_count', 'idx_store_sales_ss_sold_date_sk', 'idx_store_sales_ss_sold_time_sk', 'idx_store_sales_ss_item_sk', 'idx_store_sales_ss_customer_sk', 'idx_store_sales_ss_cdemo_sk', 'idx_store_sales_ss_hdemo_sk', 'idx_store_sales_ss_addr_sk', 'idx_store_sales_ss_store_sk', 'idx_store_sales_ss_promo_sk', 'idx_store_sales_ss_ticket_number', 'idx_store_sales_ss_quantity', 'idx_store_sales_ss_wholesale_cost', 'idx_store_sales_ss_list_price', 'idx_store_sales_ss_sales_price', 'idx_store_sales_ss_ext_discount_amt', 'idx_store_sales_ss_ext_sales_price', 'idx_store_sales_ss_ext_wholesale_cost', 'idx_store_sales_ss_ext_list_price', 'idx_store_sales_ss_ext_tax', 'idx_store_sales_ss_coupon_amt', 'idx_store_sales_ss_net_paid', 'idx_store_sales_ss_net_paid_inc_tax', 'idx_store_sales_ss_net_profit', 'idx_web_page_wp_web_page_sk', 'idx_web_page_wp_web_page_id', 'idx_web_page_wp_rec_start_date', 'idx_web_page_wp_rec_end_date', 'idx_web_page_wp_creation_date_sk', 'idx_web_page_wp_access_date_sk', 'idx_web_page_wp_autogen_flag', 'idx_web_page_wp_customer_sk', 'idx_web_page_wp_url', 'idx_web_page_wp_type', 'idx_web_page_wp_char_count', 'idx_web_page_wp_link_count', 'idx_web_page_wp_image_count', 'idx_web_page_wp_max_ad_count', 'idx_customer_c_customer_sk', 'idx_customer_c_customer_id', 'idx_customer_c_current_cdemo_sk', 'idx_customer_c_current_hdemo_sk', 'idx_customer_c_current_addr_sk', 'idx_customer_c_first_shipto_date_sk', 'idx_customer_c_first_sales_date_sk', 'idx_customer_c_salutation', 'idx_customer_c_first_name', 'idx_customer_c_last_name', 'idx_customer_c_preferred_cust_flag', 'idx_customer_c_birth_day', 'idx_customer_c_birth_month', 'idx_customer_c_birth_year', 'idx_customer_c_birth_country', 'idx_customer_c_login', 'idx_customer_c_email_address', 'idx_customer_c_last_review_date_sk', 'idx_item_i_item_sk', 'idx_item_i_item_id', 'idx_item_i_rec_start_date', 'idx_item_i_rec_end_date', 'idx_item_i_item_desc', 'idx_item_i_current_price', 'idx_item_i_wholesale_cost', 'idx_item_i_brand_id', 'idx_item_i_brand', 'idx_item_i_class_id', 'idx_item_i_class', 'idx_item_i_category_id', 'idx_item_i_category', 'idx_item_i_manufact_id', 'idx_item_i_manufact', 'idx_item_i_size', 'idx_item_i_formulation', 'idx_item_i_color', 'idx_item_i_units', 'idx_item_i_container', 'idx_item_i_manager_id', 'idx_item_i_product_name', 'idx_web_site_web_site_sk', 'idx_web_site_web_site_id', 'idx_web_site_web_rec_start_date', 'idx_web_site_web_rec_end_date', 'idx_web_site_web_name', 'idx_web_site_web_open_date_sk', 'idx_web_site_web_close_date_sk', 'idx_web_site_web_class', 'idx_web_site_web_manager', 'idx_web_site_web_mkt_id', 'idx_web_site_web_mkt_class', 'idx_web_site_web_mkt_desc', 'idx_web_site_web_market_manager', 'idx_web_site_web_company_id', 'idx_web_site_web_company_name', 'idx_web_site_web_street_number', 'idx_web_site_web_street_name', 'idx_web_site_web_street_type', 'idx_web_site_web_suite_number', 'idx_web_site_web_city', 'idx_web_site_web_county', 'idx_web_site_web_state', 'idx_web_site_web_zip', 'idx_web_site_web_country', 'idx_web_site_web_gmt_offset', 'idx_web_site_web_tax_percentage', 'idx_catalog_page_cp_catalog_page_sk', 'idx_catalog_page_cp_catalog_page_id', 'idx_catalog_page_cp_start_date_sk', 'idx_catalog_page_cp_end_date_sk', 'idx_catalog_page_cp_department', 'idx_catalog_page_cp_catalog_number', 'idx_catalog_page_cp_catalog_page_number', 'idx_catalog_page_cp_description', 'idx_catalog_page_cp_type', 'idx_customer_demographics_cd_demo_sk', 'idx_customer_demographics_cd_gender', 'idx_customer_demographics_cd_marital_status', 'idx_customer_demographics_cd_education_status', 'idx_customer_demographics_cd_purchase_estimate', 'idx_customer_demographics_cd_credit_rating', 'idx_customer_demographics_cd_dep_count', 'idx_customer_demographics_cd_dep_employed_count', 'idx_customer_demographics_cd_dep_college_count', 'idx_promotion_p_promo_sk', 'idx_promotion_p_promo_id', 'idx_promotion_p_start_date_sk', 'idx_promotion_p_end_date_sk', 'idx_promotion_p_item_sk', 'idx_promotion_p_cost', 'idx_promotion_p_response_target', 'idx_promotion_p_promo_name', 'idx_promotion_p_channel_dmail', 'idx_promotion_p_channel_email', 'idx_promotion_p_channel_catalog', 'idx_promotion_p_channel_tv', 'idx_promotion_p_channel_radio', 'idx_promotion_p_channel_press', 'idx_promotion_p_channel_event', 'idx_promotion_p_channel_demo', 'idx_promotion_p_channel_details', 'idx_promotion_p_purpose', 'idx_promotion_p_discount_active', 'idx_web_returns_wr_returned_date_sk', 'idx_web_returns_wr_returned_time_sk', 'idx_web_returns_wr_item_sk', 'idx_web_returns_wr_refunded_customer_sk', 'idx_web_returns_wr_refunded_cdemo_sk', 'idx_web_returns_wr_refunded_hdemo_sk', 'idx_web_returns_wr_refunded_addr_sk', 'idx_web_returns_wr_returning_customer_sk', 'idx_web_returns_wr_returning_cdemo_sk', 'idx_web_returns_wr_returning_hdemo_sk', 'idx_web_returns_wr_returning_addr_sk', 'idx_web_returns_wr_web_page_sk', 'idx_web_returns_wr_reason_sk', 'idx_web_returns_wr_order_number', 'idx_web_returns_wr_return_quantity', 'idx_web_returns_wr_return_amt', 'idx_web_returns_wr_return_tax', 'idx_web_returns_wr_return_amt_inc_tax', 'idx_web_returns_wr_fee', 'idx_web_returns_wr_return_ship_cost', 'idx_web_returns_wr_refunded_cash', 'idx_web_returns_wr_reversed_charge', 'idx_web_returns_wr_account_credit', 'idx_web_returns_wr_net_loss', 'idx_call_center_cc_call_center_sk', 'idx_call_center_cc_call_center_id', 'idx_call_center_cc_rec_start_date', 'idx_call_center_cc_rec_end_date', 'idx_call_center_cc_closed_date_sk', 'idx_call_center_cc_open_date_sk', 'idx_call_center_cc_name', 'idx_call_center_cc_class', 'idx_call_center_cc_employees',
                 'idx_call_center_cc_sq_ft', 'idx_call_center_cc_hours', 'idx_call_center_cc_manager', 'idx_call_center_cc_mkt_id', 'idx_call_center_cc_mkt_class', 'idx_call_center_cc_mkt_desc', 'idx_call_center_cc_market_manager', 'idx_call_center_cc_division', 'idx_call_center_cc_division_name', 'idx_call_center_cc_company', 'idx_call_center_cc_company_name', 'idx_call_center_cc_street_number', 'idx_call_center_cc_street_name', 'idx_call_center_cc_street_type', 'idx_call_center_cc_suite_number', 'idx_call_center_cc_city', 'idx_call_center_cc_county', 'idx_call_center_cc_state', 'idx_call_center_cc_zip', 'idx_call_center_cc_country', 'idx_call_center_cc_gmt_offset', 'idx_call_center_cc_tax_percentage', 'idx_date_dim_d_date_sk', 'idx_date_dim_d_date_id', 'idx_date_dim_d_date', 'idx_date_dim_d_month_seq', 'idx_date_dim_d_week_seq', 'idx_date_dim_d_quarter_seq', 'idx_date_dim_d_year', 'idx_date_dim_d_dow', 'idx_date_dim_d_moy', 'idx_date_dim_d_dom', 'idx_date_dim_d_qoy', 'idx_date_dim_d_fy_year', 'idx_date_dim_d_fy_quarter_seq', 'idx_date_dim_d_fy_week_seq', 'idx_date_dim_d_day_name', 'idx_date_dim_d_quarter_name', 'idx_date_dim_d_holiday', 'idx_date_dim_d_weekend', 'idx_date_dim_d_following_holiday', 'idx_date_dim_d_first_dom', 'idx_date_dim_d_last_dom', 'idx_date_dim_d_same_day_ly', 'idx_date_dim_d_same_day_lq', 'idx_date_dim_d_current_day', 'idx_date_dim_d_current_week', 'idx_date_dim_d_current_month', 'idx_date_dim_d_current_quarter', 'idx_date_dim_d_current_year', 'idx_web_sales_ws_sold_date_sk', 'idx_web_sales_ws_sold_time_sk', 'idx_web_sales_ws_ship_date_sk', 'idx_web_sales_ws_item_sk', 'idx_web_sales_ws_bill_customer_sk', 'idx_web_sales_ws_bill_cdemo_sk', 'idx_web_sales_ws_bill_hdemo_sk', 'idx_web_sales_ws_bill_addr_sk', 'idx_web_sales_ws_ship_customer_sk', 'idx_web_sales_ws_ship_cdemo_sk', 'idx_web_sales_ws_ship_hdemo_sk', 'idx_web_sales_ws_ship_addr_sk', 'idx_web_sales_ws_web_page_sk', 'idx_web_sales_ws_web_site_sk', 'idx_web_sales_ws_ship_mode_sk', 'idx_web_sales_ws_warehouse_sk', 'idx_web_sales_ws_promo_sk', 'idx_web_sales_ws_order_number', 'idx_web_sales_ws_quantity', 'idx_web_sales_ws_wholesale_cost', 'idx_web_sales_ws_list_price', 'idx_web_sales_ws_sales_price', 'idx_web_sales_ws_ext_discount_amt', 'idx_web_sales_ws_ext_sales_price', 'idx_web_sales_ws_ext_wholesale_cost', 'idx_web_sales_ws_ext_list_price', 'idx_web_sales_ws_ext_tax', 'idx_web_sales_ws_coupon_amt', 'idx_web_sales_ws_ext_ship_cost', 'idx_web_sales_ws_net_paid', 'idx_web_sales_ws_net_paid_inc_tax', 'idx_web_sales_ws_net_paid_inc_ship', 'idx_web_sales_ws_net_paid_inc_ship_tax', 'idx_web_sales_ws_net_profit', 'idx_customer_address_ca_address_sk', 'idx_customer_address_ca_address_id', 'idx_customer_address_ca_street_number', 'idx_customer_address_ca_street_name', 'idx_customer_address_ca_street_type', 'idx_customer_address_ca_suite_number', 'idx_customer_address_ca_city', 'idx_customer_address_ca_county', 'idx_customer_address_ca_state', 'idx_customer_address_ca_zip', 'idx_customer_address_ca_country', 'idx_customer_address_ca_gmt_offset', 'idx_customer_address_ca_location_type', 'idx_catalog_sales_cs_sold_date_sk', 'idx_catalog_sales_cs_sold_time_sk', 'idx_catalog_sales_cs_ship_date_sk', 'idx_catalog_sales_cs_bill_customer_sk', 'idx_catalog_sales_cs_bill_cdemo_sk', 'idx_catalog_sales_cs_bill_hdemo_sk', 'idx_catalog_sales_cs_bill_addr_sk', 'idx_catalog_sales_cs_ship_customer_sk', 'idx_catalog_sales_cs_ship_cdemo_sk', 'idx_catalog_sales_cs_ship_hdemo_sk', 'idx_catalog_sales_cs_ship_addr_sk', 'idx_catalog_sales_cs_call_center_sk', 'idx_catalog_sales_cs_catalog_page_sk', 'idx_catalog_sales_cs_ship_mode_sk', 'idx_catalog_sales_cs_warehouse_sk', 'idx_catalog_sales_cs_item_sk', 'idx_catalog_sales_cs_promo_sk', 'idx_catalog_sales_cs_order_number', 'idx_catalog_sales_cs_quantity', 'idx_catalog_sales_cs_wholesale_cost', 'idx_catalog_sales_cs_list_price', 'idx_catalog_sales_cs_sales_price', 'idx_catalog_sales_cs_ext_discount_amt', 'idx_catalog_sales_cs_ext_sales_price', 'idx_catalog_sales_cs_ext_wholesale_cost', 'idx_catalog_sales_cs_ext_list_price', 'idx_catalog_sales_cs_ext_tax', 'idx_catalog_sales_cs_coupon_amt', 'idx_catalog_sales_cs_ext_ship_cost', 'idx_catalog_sales_cs_net_paid', 'idx_catalog_sales_cs_net_paid_inc_tax', 'idx_catalog_sales_cs_net_paid_inc_ship', 'idx_catalog_sales_cs_net_paid_inc_ship_tax', 'idx_catalog_sales_cs_net_profit', 'idx_reason_r_reason_sk', 'idx_reason_r_reason_id', 'idx_reason_r_reason_desc', 'idx_catalog_returns_cr_returned_date_sk', 'idx_catalog_returns_cr_returned_time_sk', 'idx_catalog_returns_cr_item_sk', 'idx_catalog_returns_cr_refunded_customer_sk', 'idx_catalog_returns_cr_refunded_cdemo_sk', 'idx_catalog_returns_cr_refunded_hdemo_sk', 'idx_catalog_returns_cr_refunded_addr_sk', 'idx_catalog_returns_cr_returning_customer_sk', 'idx_catalog_returns_cr_returning_cdemo_sk', 'idx_catalog_returns_cr_returning_hdemo_sk', 'idx_catalog_returns_cr_returning_addr_sk', 'idx_catalog_returns_cr_call_center_sk', 'idx_catalog_returns_cr_catalog_page_sk', 'idx_catalog_returns_cr_ship_mode_sk', 'idx_catalog_returns_cr_warehouse_sk', 'idx_catalog_returns_cr_reason_sk', 'idx_catalog_returns_cr_order_number', 'idx_catalog_returns_cr_return_quantity', 'idx_catalog_returns_cr_return_amount', 'idx_catalog_returns_cr_return_tax', 'idx_catalog_returns_cr_return_amt_inc_tax', 'idx_catalog_returns_cr_fee', 'idx_catalog_returns_cr_return_ship_cost', 'idx_catalog_returns_cr_refunded_cash', 'idx_catalog_returns_cr_reversed_charge', 'idx_catalog_returns_cr_store_credit', 'idx_catalog_returns_cr_net_loss', 'idx_time_dim_t_time_sk', 'idx_time_dim_t_time_id', 'idx_time_dim_t_time', 'idx_time_dim_t_hour', 'idx_time_dim_t_minute', 'idx_time_dim_t_second', 'idx_time_dim_t_am_pm', 'idx_time_dim_t_shift', 'idx_time_dim_t_sub_shift', 'idx_time_dim_t_meal_time', 'idx_ship_mode_sm_ship_mode_sk', 'idx_ship_mode_sm_ship_mode_id', 'idx_ship_mode_sm_type', 'idx_ship_mode_sm_code', 'idx_ship_mode_sm_carrier', 'idx_ship_mode_sm_contract', 'idx_income_band_ib_income_band_sk', 'idx_income_band_ib_lower_bound', 'idx_income_band_ib_upper_bound', 'idx_store_s_store_sk', 'idx_store_s_store_id', 'idx_store_s_rec_start_date', 'idx_store_s_rec_end_date', 'idx_store_s_closed_date_sk', 'idx_store_s_store_name', 'idx_store_s_number_employees', 'idx_store_s_floor_space', 'idx_store_s_hours', 'idx_store_s_manager', 'idx_store_s_market_id', 'idx_store_s_geography_class', 'idx_store_s_market_desc', 'idx_store_s_market_manager', 'idx_store_s_division_id', 'idx_store_s_division_name', 'idx_store_s_company_id', 'idx_store_s_company_name', 'idx_store_s_street_number', 'idx_store_s_street_name', 'idx_store_s_street_type', 'idx_store_s_suite_number', 'idx_store_s_city', 'idx_store_s_county', 'idx_store_s_state', 'idx_store_s_zip', 'idx_store_s_country', 'idx_store_s_gmt_offset', 'idx_store_s_tax_precentage', 'idx_inventory_inv_date_sk', 'idx_inventory_inv_item_sk', 'idx_inventory_inv_warehouse_sk', 'idx_inventory_inv_quantity_on_hand', 
                 'catalog_returns_pkey', 'catalog_sales_pkey', 'inventory_pkey', 'store_returns_pkey', 'store_sales_pkey', 'web_returns_pkey', 'web_sales_pkey']
SYN_M_INDEXES = ['table0_pkey', 'table1_pkey', 'table2_pkey', 'table3_pkey', 'table4_pkey', 'table5_pkey', 'table6_pkey', 'table7_pkey', 'table8_pkey', 'table9_pkey']
SYN_S_INDEXES = ['idx_table0_col0', 'idx_table0_col1', 'idx_table0_col2', 'idx_table0_col3', 'idx_table0_col4', 'idx_table0_col5', 'idx_table0_col6', 'idx_table0_col7', 'idx_table0_col8', 'idx_table0_col9']

IDX_TO_TABLE = {
    'aka_name_pkey': 'aka_name', 'aka_title_pkey': 'aka_title', 'cast_info_pkey': 'cast_info', 'char_name_pkey': 'char_name', 'comp_cast_type_pkey': 'comp_cast_type', 'company_name_pkey': 'company_name', 'company_type_pkey': 'company_type', 'complete_cast_pkey': 'complete_cast', 'info_type_pkey': 'info_type', 'keyword_pkey': 'keyword', 'kind_type_pkey': 'kind_type', 'link_type_pkey': 'link_type', 'movie_companies_pkey': 'movie_companies', 'movie_info_idx_pkey': 'movie_info_idx', 'movie_keyword_pkey': 'movie_keyword', 'movie_link_pkey': 'movie_link', 'name_pkey': 'name', 'role_type_pkey': 'role_type', 'title_pkey': 'title', 'movie_info_pkey': 'movie_info', 'person_info_pkey': 'person_info', 'company_id_movie_companies': 'movie_companies', 'company_type_id_movie_companies': 'movie_companies', 'info_type_id_movie_info_idx': 'movie_info_idx', 'info_type_id_movie_info': 'movie_info', 'info_type_id_person_info': 'person_info', 'keyword_id_movie_keyword': 'movie_keyword', 'kind_id_aka_title': 'aka_title', 'kind_id_title': 'title', 'linked_movie_id_movie_link': 'movie_link', 'link_type_id_movie_link': 'movie_link', 'movie_id_aka_title': 'aka_title', 'movie_id_cast_info': 'cast_info', 'movie_id_complete_cast': 'complete_cast', 'movie_id_movie_companies': 'movie_companies', 'movie_id_movie_info_idx': 'movie_info_idx', 'movie_id_movie_keyword': 'movie_keyword', 'movie_id_movie_link': 'movie_link', 'movie_id_movie_info': 'movie_info', 'person_id_aka_name': 'aka_name', 'person_id_cast_info': 'cast_info', 'person_id_person_info': 'person_info', 'person_role_id_cast_info': 'cast_info', 'role_id_cast_info': 'cast_info', 
    'idx_warehouse_w_warehouse_sk': 'warehouse', 'idx_warehouse_w_warehouse_id': 'warehouse', 'idx_warehouse_w_warehouse_name': 'warehouse', 'idx_warehouse_w_warehouse_sq_ft': 'warehouse', 'idx_warehouse_w_street_number': 'warehouse', 'idx_warehouse_w_street_name': 'warehouse', 'idx_warehouse_w_street_type': 'warehouse', 'idx_warehouse_w_suite_number': 'warehouse', 'idx_warehouse_w_city': 'warehouse', 'idx_warehouse_w_county': 'warehouse', 'idx_warehouse_w_state': 'warehouse', 'idx_warehouse_w_zip': 'warehouse', 'idx_warehouse_w_country': 'warehouse', 'idx_warehouse_w_gmt_offset': 'warehouse', 'idx_store_returns_sr_returned_date_sk': 'store_returns', 'idx_store_returns_sr_return_time_sk': 'store_returns', 'idx_store_returns_sr_item_sk': 'store_returns', 'idx_store_returns_sr_customer_sk': 'store_returns', 'idx_store_returns_sr_cdemo_sk': 'store_returns', 'idx_store_returns_sr_hdemo_sk': 'store_returns', 'idx_store_returns_sr_addr_sk': 'store_returns', 'idx_store_returns_sr_store_sk': 'store_returns', 'idx_store_returns_sr_reason_sk': 'store_returns', 'idx_store_returns_sr_ticket_number': 'store_returns', 'idx_store_returns_sr_return_quantity': 'store_returns', 'idx_store_returns_sr_return_amt': 'store_returns', 'idx_store_returns_sr_return_tax': 'store_returns', 'idx_store_returns_sr_return_amt_inc_tax': 'store_returns', 'idx_store_returns_sr_fee': 'store_returns', 'idx_store_returns_sr_return_ship_cost': 'store_returns', 'idx_store_returns_sr_refunded_cash': 'store_returns', 'idx_store_returns_sr_reversed_charge': 'store_returns', 'idx_store_returns_sr_store_credit': 'store_returns', 'idx_store_returns_sr_net_loss': 'store_returns', 'idx_household_demographics_hd_demo_sk': 'household_demographics', 'idx_household_demographics_hd_income_band_sk': 'household_demographics', 'idx_household_demographics_hd_buy_potential': 'household_demographics', 'idx_household_demographics_hd_dep_count': 'household_demographics', 'idx_household_demographics_hd_vehicle_count': 'household_demographics', 'idx_store_sales_ss_sold_date_sk': 'store_sales', 'idx_store_sales_ss_sold_time_sk': 'store_sales', 'idx_store_sales_ss_item_sk': 'store_sales', 'idx_store_sales_ss_customer_sk': 'store_sales', 'idx_store_sales_ss_cdemo_sk': 'store_sales', 'idx_store_sales_ss_hdemo_sk': 'store_sales', 'idx_store_sales_ss_addr_sk': 'store_sales', 'idx_store_sales_ss_store_sk': 'store_sales', 'idx_store_sales_ss_promo_sk': 'store_sales', 'idx_store_sales_ss_ticket_number': 'store_sales', 'idx_store_sales_ss_quantity': 'store_sales', 'idx_store_sales_ss_wholesale_cost': 'store_sales', 'idx_store_sales_ss_list_price': 'store_sales', 'idx_store_sales_ss_sales_price': 'store_sales', 'idx_store_sales_ss_ext_discount_amt': 'store_sales', 'idx_store_sales_ss_ext_sales_price': 'store_sales', 'idx_store_sales_ss_ext_wholesale_cost': 'store_sales', 'idx_store_sales_ss_ext_list_price': 'store_sales', 'idx_store_sales_ss_ext_tax': 'store_sales', 'idx_store_sales_ss_coupon_amt': 'store_sales', 'idx_store_sales_ss_net_paid': 'store_sales', 'idx_store_sales_ss_net_paid_inc_tax': 'store_sales', 'idx_store_sales_ss_net_profit': 'store_sales', 'idx_web_page_wp_web_page_sk': 'web_page', 'idx_web_page_wp_web_page_id': 'web_page', 'idx_web_page_wp_rec_start_date': 'web_page', 'idx_web_page_wp_rec_end_date': 'web_page', 'idx_web_page_wp_creation_date_sk': 'web_page', 'idx_web_page_wp_access_date_sk': 'web_page', 'idx_web_page_wp_autogen_flag': 'web_page', 'idx_web_page_wp_customer_sk': 'web_page', 'idx_web_page_wp_url': 'web_page', 'idx_web_page_wp_type': 'web_page', 'idx_web_page_wp_char_count': 'web_page', 'idx_web_page_wp_link_count': 'web_page', 'idx_web_page_wp_image_count': 'web_page', 'idx_web_page_wp_max_ad_count': 'web_page', 'idx_customer_c_customer_sk': 'customer', 'idx_customer_c_customer_id': 'customer', 'idx_customer_c_current_cdemo_sk': 'customer', 'idx_customer_c_current_hdemo_sk': 'customer', 'idx_customer_c_current_addr_sk': 'customer', 'idx_customer_c_first_shipto_date_sk': 'customer', 'idx_customer_c_first_sales_date_sk': 'customer', 'idx_customer_c_salutation': 'customer', 'idx_customer_c_first_name': 'customer', 'idx_customer_c_last_name': 'customer', 'idx_customer_c_preferred_cust_flag': 'customer', 'idx_customer_c_birth_day': 'customer', 'idx_customer_c_birth_month': 'customer', 'idx_customer_c_birth_year': 'customer', 'idx_customer_c_birth_country': 'customer', 'idx_customer_c_login': 'customer', 'idx_customer_c_email_address': 'customer', 'idx_customer_c_last_review_date_sk': 'customer', 'idx_item_i_item_sk': 'item', 'idx_item_i_item_id': 'item', 'idx_item_i_rec_start_date': 'item', 'idx_item_i_rec_end_date': 'item', 'idx_item_i_item_desc': 'item', 'idx_item_i_current_price': 'item', 'idx_item_i_wholesale_cost': 'item', 'idx_item_i_brand_id': 'item', 'idx_item_i_brand': 'item', 'idx_item_i_class_id': 'item', 'idx_item_i_class': 'item', 'idx_item_i_category_id': 'item', 'idx_item_i_category': 'item', 'idx_item_i_manufact_id': 'item', 'idx_item_i_manufact': 'item', 'idx_item_i_size': 'item', 'idx_item_i_formulation': 'item', 'idx_item_i_color': 'item', 'idx_item_i_units': 'item', 'idx_item_i_container': 'item', 'idx_item_i_manager_id': 'item', 'idx_item_i_product_name': 'item', 'idx_web_site_web_site_sk': 'web_site', 'idx_web_site_web_site_id': 'web_site', 'idx_web_site_web_rec_start_date': 'web_site', 'idx_web_site_web_rec_end_date': 'web_site', 'idx_web_site_web_name': 'web_site', 'idx_web_site_web_open_date_sk': 'web_site', 'idx_web_site_web_close_date_sk': 'web_site', 'idx_web_site_web_class': 'web_site', 'idx_web_site_web_manager': 'web_site', 'idx_web_site_web_mkt_id': 'web_site', 'idx_web_site_web_mkt_class': 'web_site', 'idx_web_site_web_mkt_desc': 'web_site', 'idx_web_site_web_market_manager': 'web_site', 'idx_web_site_web_company_id': 'web_site', 'idx_web_site_web_company_name': 'web_site', 'idx_web_site_web_street_number': 'web_site', 'idx_web_site_web_street_name': 'web_site', 'idx_web_site_web_street_type': 'web_site', 'idx_web_site_web_suite_number': 'web_site', 'idx_web_site_web_city': 'web_site', 'idx_web_site_web_county': 'web_site', 'idx_web_site_web_state': 'web_site', 'idx_web_site_web_zip': 'web_site', 'idx_web_site_web_country': 'web_site', 'idx_web_site_web_gmt_offset': 'web_site', 'idx_web_site_web_tax_percentage': 'web_site', 'idx_catalog_page_cp_catalog_page_sk': 'catalog_page', 'idx_catalog_page_cp_catalog_page_id': 'catalog_page', 'idx_catalog_page_cp_start_date_sk': 'catalog_page', 'idx_catalog_page_cp_end_date_sk': 'catalog_page', 'idx_catalog_page_cp_department': 'catalog_page', 'idx_catalog_page_cp_catalog_number': 'catalog_page', 'idx_catalog_page_cp_catalog_page_number': 'catalog_page', 'idx_catalog_page_cp_description': 'catalog_page', 'idx_catalog_page_cp_type': 'catalog_page', 'idx_customer_demographics_cd_demo_sk': 'customer_demographics', 'idx_customer_demographics_cd_gender': 'customer_demographics', 'idx_customer_demographics_cd_marital_status': 'customer_demographics', 'idx_customer_demographics_cd_education_status': 'customer_demographics', 'idx_customer_demographics_cd_purchase_estimate': 'customer_demographics', 'idx_customer_demographics_cd_credit_rating': 'customer_demographics', 'idx_customer_demographics_cd_dep_count': 'customer_demographics', 'idx_customer_demographics_cd_dep_employed_count': 'customer_demographics', 'idx_customer_demographics_cd_dep_college_count': 'customer_demographics', 'idx_promotion_p_promo_sk': 'promotion', 'idx_promotion_p_promo_id': 'promotion', 'idx_promotion_p_start_date_sk': 'promotion', 'idx_promotion_p_end_date_sk': 'promotion', 'idx_promotion_p_item_sk': 'promotion', 'idx_promotion_p_cost': 'promotion', 'idx_promotion_p_response_target': 'promotion', 'idx_promotion_p_promo_name': 'promotion', 'idx_promotion_p_channel_dmail': 'promotion', 'idx_promotion_p_channel_email': 'promotion', 'idx_promotion_p_channel_catalog': 'promotion', 'idx_promotion_p_channel_tv': 'promotion', 'idx_promotion_p_channel_radio': 'promotion', 'idx_promotion_p_channel_press': 'promotion', 'idx_promotion_p_channel_event': 'promotion', 'idx_promotion_p_channel_demo': 'promotion', 'idx_promotion_p_channel_details': 'promotion', 'idx_promotion_p_purpose': 'promotion', 'idx_promotion_p_discount_active': 'promotion', 'idx_web_returns_wr_returned_date_sk': 'web_returns', 'idx_web_returns_wr_returned_time_sk': 'web_returns', 'idx_web_returns_wr_item_sk': 'web_returns', 'idx_web_returns_wr_refunded_customer_sk': 'web_returns', 'idx_web_returns_wr_refunded_cdemo_sk': 'web_returns', 'idx_web_returns_wr_refunded_hdemo_sk': 'web_returns', 'idx_web_returns_wr_refunded_addr_sk': 'web_returns', 'idx_web_returns_wr_returning_customer_sk': 'web_returns', 'idx_web_returns_wr_returning_cdemo_sk': 'web_returns', 'idx_web_returns_wr_returning_hdemo_sk': 'web_returns', 'idx_web_returns_wr_returning_addr_sk': 'web_returns', 'idx_web_returns_wr_web_page_sk': 'web_returns', 'idx_web_returns_wr_reason_sk': 'web_returns', 'idx_web_returns_wr_order_number': 'web_returns', 'idx_web_returns_wr_return_quantity': 'web_returns', 'idx_web_returns_wr_return_amt': 'web_returns', 'idx_web_returns_wr_return_tax': 'web_returns', 'idx_web_returns_wr_return_amt_inc_tax': 'web_returns', 'idx_web_returns_wr_fee': 'web_returns', 'idx_web_returns_wr_return_ship_cost': 'web_returns', 'idx_web_returns_wr_refunded_cash': 'web_returns', 'idx_web_returns_wr_reversed_charge': 'web_returns', 'idx_web_returns_wr_account_credit': 'web_returns', 'idx_web_returns_wr_net_loss': 'web_returns', 'idx_call_center_cc_call_center_sk': 'call_center', 'idx_call_center_cc_call_center_id': 'call_center', 'idx_call_center_cc_rec_start_date': 'call_center', 'idx_call_center_cc_rec_end_date': 'call_center', 'idx_call_center_cc_closed_date_sk': 'call_center', 'idx_call_center_cc_open_date_sk': 'call_center', 'idx_call_center_cc_name': 'call_center', 'idx_call_center_cc_class': 'call_center', 'idx_call_center_cc_employees': 
    'call_center', 'idx_call_center_cc_sq_ft': 'call_center', 'idx_call_center_cc_hours': 'call_center', 'idx_call_center_cc_manager': 'call_center', 'idx_call_center_cc_mkt_id': 'call_center', 'idx_call_center_cc_mkt_class': 'call_center', 'idx_call_center_cc_mkt_desc': 'call_center', 'idx_call_center_cc_market_manager': 'call_center', 'idx_call_center_cc_division': 'call_center', 'idx_call_center_cc_division_name': 'call_center', 'idx_call_center_cc_company': 'call_center', 'idx_call_center_cc_company_name': 'call_center', 'idx_call_center_cc_street_number': 'call_center', 'idx_call_center_cc_street_name': 'call_center', 'idx_call_center_cc_street_type': 'call_center', 'idx_call_center_cc_suite_number': 'call_center', 'idx_call_center_cc_city': 'call_center', 'idx_call_center_cc_county': 'call_center', 'idx_call_center_cc_state': 'call_center', 'idx_call_center_cc_zip': 'call_center', 'idx_call_center_cc_country': 'call_center', 'idx_call_center_cc_gmt_offset': 'call_center', 'idx_call_center_cc_tax_percentage': 'call_center', 'idx_date_dim_d_date_sk': 'date_dim', 'idx_date_dim_d_date_id': 'date_dim', 'idx_date_dim_d_date': 'date_dim', 'idx_date_dim_d_month_seq': 'date_dim', 'idx_date_dim_d_week_seq': 'date_dim', 'idx_date_dim_d_quarter_seq': 'date_dim', 'idx_date_dim_d_year': 'date_dim', 'idx_date_dim_d_dow': 'date_dim', 'idx_date_dim_d_moy': 'date_dim', 'idx_date_dim_d_dom': 'date_dim', 'idx_date_dim_d_qoy': 'date_dim', 'idx_date_dim_d_fy_year': 'date_dim', 'idx_date_dim_d_fy_quarter_seq': 'date_dim', 'idx_date_dim_d_fy_week_seq': 'date_dim', 'idx_date_dim_d_day_name': 'date_dim', 'idx_date_dim_d_quarter_name': 'date_dim', 'idx_date_dim_d_holiday': 'date_dim', 'idx_date_dim_d_weekend': 'date_dim', 'idx_date_dim_d_following_holiday': 'date_dim', 'idx_date_dim_d_first_dom': 'date_dim', 'idx_date_dim_d_last_dom': 'date_dim', 'idx_date_dim_d_same_day_ly': 'date_dim', 'idx_date_dim_d_same_day_lq': 'date_dim', 'idx_date_dim_d_current_day': 'date_dim', 'idx_date_dim_d_current_week': 'date_dim', 'idx_date_dim_d_current_month': 'date_dim', 'idx_date_dim_d_current_quarter': 'date_dim', 'idx_date_dim_d_current_year': 'date_dim', 'idx_web_sales_ws_sold_date_sk': 'web_sales', 'idx_web_sales_ws_sold_time_sk': 'web_sales', 'idx_web_sales_ws_ship_date_sk': 'web_sales', 'idx_web_sales_ws_item_sk': 'web_sales', 'idx_web_sales_ws_bill_customer_sk': 'web_sales', 'idx_web_sales_ws_bill_cdemo_sk': 'web_sales', 'idx_web_sales_ws_bill_hdemo_sk': 'web_sales', 'idx_web_sales_ws_bill_addr_sk': 'web_sales', 'idx_web_sales_ws_ship_customer_sk': 'web_sales', 'idx_web_sales_ws_ship_cdemo_sk': 'web_sales', 'idx_web_sales_ws_ship_hdemo_sk': 'web_sales', 'idx_web_sales_ws_ship_addr_sk': 'web_sales', 'idx_web_sales_ws_web_page_sk': 'web_sales', 'idx_web_sales_ws_web_site_sk': 'web_sales', 'idx_web_sales_ws_ship_mode_sk': 'web_sales', 'idx_web_sales_ws_warehouse_sk': 'web_sales', 'idx_web_sales_ws_promo_sk': 'web_sales', 'idx_web_sales_ws_order_number': 'web_sales', 'idx_web_sales_ws_quantity': 'web_sales', 'idx_web_sales_ws_wholesale_cost': 'web_sales', 'idx_web_sales_ws_list_price': 'web_sales', 'idx_web_sales_ws_sales_price': 'web_sales', 'idx_web_sales_ws_ext_discount_amt': 'web_sales', 'idx_web_sales_ws_ext_sales_price': 'web_sales', 'idx_web_sales_ws_ext_wholesale_cost': 'web_sales', 'idx_web_sales_ws_ext_list_price': 'web_sales', 'idx_web_sales_ws_ext_tax': 'web_sales', 'idx_web_sales_ws_coupon_amt': 'web_sales', 'idx_web_sales_ws_ext_ship_cost': 'web_sales', 'idx_web_sales_ws_net_paid': 'web_sales', 'idx_web_sales_ws_net_paid_inc_tax': 'web_sales', 'idx_web_sales_ws_net_paid_inc_ship': 'web_sales', 'idx_web_sales_ws_net_paid_inc_ship_tax': 'web_sales', 'idx_web_sales_ws_net_profit': 'web_sales', 'idx_customer_address_ca_address_sk': 'customer_address', 'idx_customer_address_ca_address_id': 'customer_address', 'idx_customer_address_ca_street_number': 'customer_address', 'idx_customer_address_ca_street_name': 'customer_address', 'idx_customer_address_ca_street_type': 'customer_address', 'idx_customer_address_ca_suite_number': 'customer_address', 'idx_customer_address_ca_city': 'customer_address', 'idx_customer_address_ca_county': 'customer_address', 'idx_customer_address_ca_state': 'customer_address', 'idx_customer_address_ca_zip': 'customer_address', 'idx_customer_address_ca_country': 'customer_address', 'idx_customer_address_ca_gmt_offset': 'customer_address', 'idx_customer_address_ca_location_type': 'customer_address', 'idx_catalog_sales_cs_sold_date_sk': 'catalog_sales', 'idx_catalog_sales_cs_sold_time_sk': 'catalog_sales', 'idx_catalog_sales_cs_ship_date_sk': 'catalog_sales', 'idx_catalog_sales_cs_bill_customer_sk': 'catalog_sales', 'idx_catalog_sales_cs_bill_cdemo_sk': 'catalog_sales', 'idx_catalog_sales_cs_bill_hdemo_sk': 'catalog_sales', 'idx_catalog_sales_cs_bill_addr_sk': 'catalog_sales', 'idx_catalog_sales_cs_ship_customer_sk': 'catalog_sales', 'idx_catalog_sales_cs_ship_cdemo_sk': 'catalog_sales', 'idx_catalog_sales_cs_ship_hdemo_sk': 'catalog_sales', 'idx_catalog_sales_cs_ship_addr_sk': 'catalog_sales', 'idx_catalog_sales_cs_call_center_sk': 'catalog_sales', 'idx_catalog_sales_cs_catalog_page_sk': 'catalog_sales', 'idx_catalog_sales_cs_ship_mode_sk': 'catalog_sales', 'idx_catalog_sales_cs_warehouse_sk': 'catalog_sales', 'idx_catalog_sales_cs_item_sk': 'catalog_sales', 'idx_catalog_sales_cs_promo_sk': 'catalog_sales', 'idx_catalog_sales_cs_order_number': 'catalog_sales', 'idx_catalog_sales_cs_quantity': 'catalog_sales', 'idx_catalog_sales_cs_wholesale_cost': 'catalog_sales', 'idx_catalog_sales_cs_list_price': 'catalog_sales', 'idx_catalog_sales_cs_sales_price': 'catalog_sales', 'idx_catalog_sales_cs_ext_discount_amt': 'catalog_sales', 'idx_catalog_sales_cs_ext_sales_price': 'catalog_sales', 'idx_catalog_sales_cs_ext_wholesale_cost': 'catalog_sales', 'idx_catalog_sales_cs_ext_list_price': 'catalog_sales', 'idx_catalog_sales_cs_ext_tax': 'catalog_sales', 'idx_catalog_sales_cs_coupon_amt': 'catalog_sales', 'idx_catalog_sales_cs_ext_ship_cost': 'catalog_sales', 'idx_catalog_sales_cs_net_paid': 'catalog_sales', 'idx_catalog_sales_cs_net_paid_inc_tax': 'catalog_sales', 'idx_catalog_sales_cs_net_paid_inc_ship': 'catalog_sales', 'idx_catalog_sales_cs_net_paid_inc_ship_tax': 'catalog_sales', 'idx_catalog_sales_cs_net_profit': 'catalog_sales', 'idx_reason_r_reason_sk': 'reason', 'idx_reason_r_reason_id': 'reason', 'idx_reason_r_reason_desc': 'reason', 'idx_catalog_returns_cr_returned_date_sk': 'catalog_returns', 'idx_catalog_returns_cr_returned_time_sk': 'catalog_returns', 'idx_catalog_returns_cr_item_sk': 'catalog_returns', 'idx_catalog_returns_cr_refunded_customer_sk': 'catalog_returns', 'idx_catalog_returns_cr_refunded_cdemo_sk': 'catalog_returns', 'idx_catalog_returns_cr_refunded_hdemo_sk': 'catalog_returns', 'idx_catalog_returns_cr_refunded_addr_sk': 'catalog_returns', 'idx_catalog_returns_cr_returning_customer_sk': 'catalog_returns', 'idx_catalog_returns_cr_returning_cdemo_sk': 'catalog_returns', 'idx_catalog_returns_cr_returning_hdemo_sk': 'catalog_returns', 'idx_catalog_returns_cr_returning_addr_sk': 'catalog_returns', 'idx_catalog_returns_cr_call_center_sk': 'catalog_returns', 'idx_catalog_returns_cr_catalog_page_sk': 'catalog_returns', 'idx_catalog_returns_cr_ship_mode_sk': 'catalog_returns', 'idx_catalog_returns_cr_warehouse_sk': 'catalog_returns', 'idx_catalog_returns_cr_reason_sk': 'catalog_returns', 'idx_catalog_returns_cr_order_number': 'catalog_returns', 'idx_catalog_returns_cr_return_quantity': 'catalog_returns', 'idx_catalog_returns_cr_return_amount': 'catalog_returns', 'idx_catalog_returns_cr_return_tax': 'catalog_returns', 'idx_catalog_returns_cr_return_amt_inc_tax': 'catalog_returns', 'idx_catalog_returns_cr_fee': 'catalog_returns', 'idx_catalog_returns_cr_return_ship_cost': 'catalog_returns', 'idx_catalog_returns_cr_refunded_cash': 'catalog_returns', 'idx_catalog_returns_cr_reversed_charge': 'catalog_returns', 'idx_catalog_returns_cr_store_credit': 'catalog_returns', 'idx_catalog_returns_cr_net_loss': 'catalog_returns', 'idx_time_dim_t_time_sk': 'time_dim', 'idx_time_dim_t_time_id': 'time_dim', 'idx_time_dim_t_time': 'time_dim', 'idx_time_dim_t_hour': 'time_dim', 'idx_time_dim_t_minute': 'time_dim', 'idx_time_dim_t_second': 'time_dim', 'idx_time_dim_t_am_pm': 'time_dim', 'idx_time_dim_t_shift': 'time_dim', 'idx_time_dim_t_sub_shift': 'time_dim', 'idx_time_dim_t_meal_time': 'time_dim', 'idx_ship_mode_sm_ship_mode_sk': 'ship_mode', 'idx_ship_mode_sm_ship_mode_id': 'ship_mode', 'idx_ship_mode_sm_type': 'ship_mode', 'idx_ship_mode_sm_code': 'ship_mode', 'idx_ship_mode_sm_carrier': 'ship_mode', 'idx_ship_mode_sm_contract': 'ship_mode', 'idx_income_band_ib_income_band_sk': 'income_band', 'idx_income_band_ib_lower_bound': 'income_band', 'idx_income_band_ib_upper_bound': 'income_band', 'idx_store_s_store_sk': 'store', 'idx_store_s_store_id': 'store', 'idx_store_s_rec_start_date': 'store', 'idx_store_s_rec_end_date': 'store', 'idx_store_s_closed_date_sk': 'store', 'idx_store_s_store_name': 'store', 'idx_store_s_number_employees': 'store', 'idx_store_s_floor_space': 'store', 'idx_store_s_hours': 'store', 'idx_store_s_manager': 'store', 'idx_store_s_market_id': 'store', 'idx_store_s_geography_class': 'store', 'idx_store_s_market_desc': 'store', 'idx_store_s_market_manager': 'store', 'idx_store_s_division_id': 'store', 'idx_store_s_division_name': 'store', 'idx_store_s_company_id': 'store', 'idx_store_s_company_name': 'store', 'idx_store_s_street_number': 'store', 'idx_store_s_street_name': 'store', 'idx_store_s_street_type': 'store', 'idx_store_s_suite_number': 'store', 'idx_store_s_city': 'store', 'idx_store_s_county': 'store', 'idx_store_s_state': 'store', 'idx_store_s_zip': 'store', 'idx_store_s_country': 'store', 'idx_store_s_gmt_offset': 'store', 'idx_store_s_tax_precentage': 'store', 'idx_inventory_inv_date_sk': 'inventory', 'idx_inventory_inv_item_sk': 'inventory', 
    'idx_inventory_inv_warehouse_sk': 'inventory', 'idx_inventory_inv_quantity_on_hand': 'inventory', 'catalog_returns_pkey': 'catalog_returns', 'catalog_sales_pkey': 'catalog_sales', 'inventory_pkey': 'inventory', 'store_returns_pkey': 'store_returns', 'store_sales_pkey': 'store_sales', 'web_returns_pkey': 'web_returns', 'web_sales_pkey' : 'web_sales', 'table0_pkey': 'table0', 'table1_pkey': 'table1', 'table2_pkey': 'table2', 'table3_pkey': 'table3', 'table4_pkey': 'table4', 'table5_pkey': 'table5', 'table6_pkey': 'table6', 'table7_pkey': 'table7', 'table8_pkey': 'table8', 'table9_pkey': 'table9', 'idx_table0_col0': 'table0', 'idx_table0_col1': 'table0', 'idx_table0_col2': 'table0', 'idx_table0_col3': 'table0', 'idx_table0_col4': 'table0', 'idx_table0_col5': 'table0', 'idx_table0_col6': 'table0', 'idx_table0_col7': 'table0', 'idx_table0_col8': 'table0', 'idx_table0_col9': 'table0'}


IMDB_SMALL_COLS = {
        'title': ['id', 'kind_id', 'production_year'],
        'cast_info': ['movie_id', 'role_id'],
        'movie_companies': ['company_id', 'company_type_id', 'movie_id'],
        'movie_info': ['movie_id','info_type_id'],
        'movie_info_idx': ['info_type_id', 'movie_id'],
        'movie_keyword': ['keyword_id', 'movie_id'],
}

IMDB_MEDIUM_COLS = {
        'title': ['id', 'kind_id', 'title', 'production_year', 'episode_nr'],
        'aka_title': ['movie_id'],
        'cast_info': ['movie_id', 'note','role_id'],
        'complete_cast': ['subject_id', 'movie_id'],
        'movie_companies': ['company_id', 'company_type_id', 'movie_id', 'note'],
        'movie_info': ['movie_id','info_type_id', 'info', 'note'],
        'movie_info_idx': ['info_type_id', 'movie_id', 'info'],
        'movie_keyword': ['keyword_id', 'movie_id'],
        'movie_link': ['link_type_id', 'movie_id'],
        'kind_type': ['id', 'kind'],
        'comp_cast_type': ['id', 'kind'],
        'company_name': ['id', 'country_code', 'name'],
        'company_type': ['id', 'kind'],
        'info_type': ['id', 'info'],
        'keyword': ['id', 'keyword'],
        'link_type': ['id', 'link'],
}

IMDB_LARGE_COLS = {
        'name': ['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'] ,
        'movie_companies': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'] ,
        'aka_name': ['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'] ,
        'movie_info': ['id', 'movie_id', 'info_type_id', 'info', 'note'] ,
        'movie_keyword': ['id', 'movie_id', 'keyword_id'] ,
        'person_info': ['id', 'person_id', 'info_type_id', 'info', 'note'] ,
        'comp_cast_type': ['id', 'kind'] ,
        'complete_cast': ['id', 'movie_id', 'subject_id', 'status_id'] ,
        'char_name': ['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf', 'surname_pcode', 'md5sum'] ,
        'movie_link': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'] ,
        'company_type': ['id', 'kind'] ,
        'cast_info': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'] ,
        'info_type': ['id', 'info'] ,
        'company_name': ['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'] ,
        'aka_title': ['id', 'movie_id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'note', 'md5sum'] ,
        'kind_type': ['id', 'kind'] ,
        'role_type': ['id', 'role'] ,
        'movie_info_idx': ['id', 'movie_id', 'info_type_id', 'info', 'note'] ,
        'keyword': ['id', 'keyword', 'phonetic_code'] ,
        'link_type': ['id', 'link'] ,
        'title': ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'] ,
}


TPCDS_BENCH_COLS = {
	'catalog_returns': ['cr_return_amount',  'cr_order_number',  'cr_returning_addr_sk',  'cr_call_center_sk',  'cr_net_loss',  'cr_reversed_charge',  'cr_store_credit',  'cr_catalog_page_sk',  'cr_return_quantity',  'cr_returned_date_sk',  'cr_returning_customer_sk',  'cr_return_amt_inc_tax',  'cr_refunded_addr_sk',  'cr_item_sk',  'cr_refunded_cash'],
        'customer': ['c_first_sales_date_sk',  'c_current_cdemo_sk',  'c_last_review_date_sk',  'c_current_hdemo_sk',  'c_customer_id',  'c_birth_month',  'c_birth_year',  'c_first_shipto_date_sk',  'c_birth_day',  'c_salutation',  'c_login',  'c_email_address',  'c_customer_sk',  'c_current_addr_sk',  'c_last_name',  'c_preferred_cust_flag',  'c_birth_country',  'c_first_name'],
        'customer_address': ['ca_street_name',  'ca_gmt_offset',  'ca_zip',  'ca_location_type',  'ca_suite_number',  'ca_country',  'ca_street_number',  'ca_city',  'ca_county',  'ca_address_sk',  'ca_street_type',  'ca_state'],
        'catalog_sales': ['cs_sales_price',  'cs_ext_sales_price',  'cs_warehouse_sk',  'cs_ext_discount_amt',  'cs_net_profit',  'cs_bill_cdemo_sk',  'cs_promo_sk',  'cs_item_sk',  'cs_bill_hdemo_sk',  'cs_ship_customer_sk',  'cs_quantity',  'cs_wholesale_cost',  'cs_ext_ship_cost',  'cs_coupon_amt',  'cs_ext_list_price',  'cs_ship_cdemo_sk',  'cs_sold_time_sk',  'cs_net_paid',  'cs_ship_addr_sk',  'cs_ext_wholesale_cost',  'cs_list_price',  'cs_order_number',  'cs_catalog_page_sk',  'cs_bill_customer_sk',  'cs_sold_date_sk',  'cs_ship_date_sk',  'cs_ship_mode_sk',  'cs_call_center_sk',  'cs_bill_addr_sk'],
        'warehouse': ['w_warehouse_name',  'w_warehouse_sq_ft',  'w_country',  'w_state',  'w_county',  'w_warehouse_sk',  'w_city'],
        'store_returns': ['sr_reason_sk',  'sr_returned_date_sk',  'sr_return_amt',  'sr_item_sk',  'sr_customer_sk',  'sr_return_quantity',  'sr_cdemo_sk',  'sr_store_sk',  'sr_ticket_number',  'sr_net_loss'],
        'customer_demographics': ['cd_dep_count',  'cd_purchase_estimate',  'cd_credit_rating',  'cd_dep_employed_count',  'cd_gender',  'cd_marital_status',  'cd_education_status',  'cd_demo_sk',  'cd_dep_college_count'],
        'web_returns': ['wr_fee',  'wr_returned_date_sk',  'wr_reason_sk',  'wr_web_page_sk',  'wr_item_sk',  'wr_returning_cdemo_sk',  'wr_return_quantity',  'wr_return_amt',  'wr_refunded_addr_sk',  'wr_order_number',  'wr_returning_addr_sk',  'wr_net_loss',  'wr_refunded_cash',  'wr_refunded_cdemo_sk',  'wr_returning_customer_sk'],
        'inventory': ['inv_quantity_on_hand',  'inv_item_sk',  'inv_warehouse_sk',  'inv_date_sk'],
        'web_sales': ['ws_web_page_sk',  'ws_bill_addr_sk',  'ws_item_sk',  'ws_ext_discount_amt',  'ws_ext_sales_price',  'ws_net_paid',  'ws_ext_wholesale_cost',  'ws_wholesale_cost',  'ws_ship_hdemo_sk',  'ws_ship_customer_sk',  'ws_ship_addr_sk',  'ws_bill_customer_sk',  'ws_ship_date_sk',  'ws_net_profit',  'ws_sold_time_sk',  'ws_bill_cdemo_sk',  'ws_warehouse_sk',  'ws_sales_price',  'ws_sold_date_sk',  'ws_order_number',  'ws_promo_sk',  'ws_list_price',  'ws_web_site_sk',  'ws_ext_ship_cost',  'ws_ship_cdemo_sk',  'ws_ship_mode_sk',  'ws_quantity',  'ws_ext_list_price'],
        'date_dim': ['d_date',  'd_moy',  'd_dow',  'd_dom',  'd_qoy',  'd_quarter_name',  'd_day_name',  'd_date_sk',  'd_month_seq',  'd_week_seq',  'd_year'],
        'store_sales': ['ss_cdemo_sk',  'ss_coupon_amt',  'ss_ext_list_price',  'ss_ext_sales_price',  'ss_item_sk',  'ss_store_sk',  'ss_ext_discount_amt',  'ss_sold_time_sk',  'ss_sold_date_sk',  'ss_ext_tax',  'ss_customer_sk',  'ss_net_profit',  'ss_ext_wholesale_cost',  'ss_wholesale_cost',  'ss_ticket_number',  'ss_hdemo_sk',  'ss_addr_sk',  'ss_promo_sk',  'ss_list_price',  'ss_sales_price',  'ss_net_paid',  'ss_quantity'],
        'item': ['i_manager_id',  'i_wholesale_cost',  'i_brand',  'i_manufact',  'i_size',  'i_category',  'i_item_sk',  'i_class',  'i_current_price',  'i_brand_id',  'i_item_desc',  'i_category_id',  'i_item_id',  'i_product_name',  'i_color',  'i_manufact_id',  'i_units',  'i_class_id'],
        'store': ['s_market_id',  's_zip',  's_company_name',  's_number_employees',  's_company_id',  's_store_id',  's_store_name',  's_street_type',  's_state',  's_county',  's_suite_number',  's_street_name',  's_gmt_offset',  's_street_number',  's_store_sk',  's_city'],
        'promotion': ['p_channel_event',  'p_channel_tv',  'p_channel_email',  'p_promo_sk',  'p_channel_dmail'],
        'catalog_page': ['cp_catalog_page_sk', 'cp_catalog_page_id'],
        'web_page': ['wp_char_count', 'wp_web_page_sk'],
        'ship_mode': ['sm_type', 'sm_ship_mode_sk', 'sm_carrier'],
        'time_dim': ['t_time', 't_minute', 't_time_sk', 't_hour', 't_meal_time'],
        'household_demographics': ['hd_dep_count',  'hd_demo_sk',  'hd_vehicle_count',  'hd_buy_potential',  'hd_income_band_sk'],
        'reason': ['r_reason_desc', 'r_reason_sk'],
        'web_site': ['web_name', 'web_company_name', 'web_site_id', 'web_site_sk'],
        'income_band': ['ib_income_band_sk', 'ib_lower_bound', 'ib_upper_bound'],
        'call_center': ['cc_call_center_sk',  'cc_name',  'cc_call_center_id',  'cc_county',  'cc_manager']
}

TPCDS_LARGE_COLS = {
        'warehouse': ['w_warehouse_sk', 'w_warehouse_id', 'w_warehouse_name', 'w_warehouse_sq_ft', 'w_street_number', 'w_street_name', 'w_street_type', 'w_suite_number', 'w_city', 'w_county', 'w_state', 'w_zip', 'w_country', 'w_gmt_offset'],
        'store_returns': ['sr_returned_date_sk', 'sr_return_time_sk', 'sr_item_sk', 'sr_customer_sk', 'sr_cdemo_sk', 'sr_hdemo_sk', 'sr_addr_sk', 'sr_store_sk', 'sr_reason_sk', 'sr_ticket_number', 'sr_return_quantity', 'sr_return_amt', 'sr_return_tax', 'sr_return_amt_inc_tax', 'sr_fee', 'sr_return_ship_cost', 'sr_refunded_cash', 'sr_reversed_charge', 'sr_store_credit', 'sr_net_loss'],
        'household_demographics': ['hd_demo_sk', 'hd_income_band_sk', 'hd_buy_potential', 'hd_dep_count', 'hd_vehicle_count'],
        'store_sales': ['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk', 'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk', 'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price', 'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax', 'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'],
        'web_page': ['wp_web_page_sk', 'wp_web_page_id', 'wp_rec_start_date', 'wp_rec_end_date', 'wp_creation_date_sk', 'wp_access_date_sk', 'wp_autogen_flag', 'wp_customer_sk', 'wp_url', 'wp_type', 'wp_char_count', 'wp_link_count', 'wp_image_count', 'wp_max_ad_count'],
        'customer': ['c_customer_sk', 'c_customer_id', 'c_current_cdemo_sk', 'c_current_hdemo_sk', 'c_current_addr_sk', 'c_first_shipto_date_sk', 'c_first_sales_date_sk', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date_sk'],
        'item': ['i_item_sk', 'i_item_id', 'i_rec_start_date', 'i_rec_end_date', 'i_item_desc', 'i_current_price', 'i_wholesale_cost', 'i_brand_id', 'i_brand', 'i_class_id', 'i_class', 'i_category_id', 'i_category', 'i_manufact_id', 'i_manufact', 'i_size', 'i_formulation', 'i_color', 'i_units', 'i_container', 'i_manager_id', 'i_product_name'],
        'web_site': ['web_site_sk', 'web_site_id', 'web_rec_start_date', 'web_rec_end_date', 'web_name', 'web_open_date_sk', 'web_close_date_sk', 'web_class', 'web_manager', 'web_mkt_id', 'web_mkt_class', 'web_mkt_desc', 'web_market_manager', 'web_company_id', 'web_company_name', 'web_street_number', 'web_street_name', 'web_street_type', 'web_suite_number', 'web_city', 'web_county', 'web_state', 'web_zip', 'web_country', 'web_gmt_offset', 'web_tax_percentage'],
        'catalog_page': ['cp_catalog_page_sk', 'cp_catalog_page_id', 'cp_start_date_sk', 'cp_end_date_sk', 'cp_department', 'cp_catalog_number', 'cp_catalog_page_number', 'cp_description', 'cp_type'],
        'customer_demographics': ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'],
        'promotion': ['p_promo_sk', 'p_promo_id', 'p_start_date_sk', 'p_end_date_sk', 'p_item_sk', 'p_cost', 'p_response_target', 'p_promo_name', 'p_channel_dmail', 'p_channel_email', 'p_channel_catalog', 'p_channel_tv', 'p_channel_radio', 'p_channel_press', 'p_channel_event', 'p_channel_demo', 'p_channel_details', 'p_purpose', 'p_discount_active'],
        'web_returns': ['wr_returned_date_sk', 'wr_returned_time_sk', 'wr_item_sk', 'wr_refunded_customer_sk', 'wr_refunded_cdemo_sk', 'wr_refunded_hdemo_sk', 'wr_refunded_addr_sk', 'wr_returning_customer_sk', 'wr_returning_cdemo_sk', 'wr_returning_hdemo_sk', 'wr_returning_addr_sk', 'wr_web_page_sk', 'wr_reason_sk', 'wr_order_number', 'wr_return_quantity', 'wr_return_amt', 'wr_return_tax', 'wr_return_amt_inc_tax', 'wr_fee', 'wr_return_ship_cost', 'wr_refunded_cash', 'wr_reversed_charge', 'wr_account_credit', 'wr_net_loss'],
        'call_center': ['cc_call_center_sk', 'cc_call_center_id', 'cc_rec_start_date', 'cc_rec_end_date', 'cc_closed_date_sk', 'cc_open_date_sk', 'cc_name', 'cc_class', 'cc_employees', 'cc_sq_ft', 'cc_hours', 'cc_manager', 'cc_mkt_id', 'cc_mkt_class', 'cc_mkt_desc', 'cc_market_manager', 'cc_division', 'cc_division_name', 'cc_company', 'cc_company_name', 'cc_street_number', 'cc_street_name', 'cc_street_type', 'cc_suite_number', 'cc_city', 'cc_county', 'cc_state', 'cc_zip', 'cc_country', 'cc_gmt_offset', 'cc_tax_percentage'],
        'date_dim': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'],
        'web_sales': ['ws_sold_date_sk', 'ws_sold_time_sk', 'ws_ship_date_sk', 'ws_item_sk', 'ws_bill_customer_sk', 'ws_bill_cdemo_sk', 'ws_bill_hdemo_sk', 'ws_bill_addr_sk', 'ws_ship_customer_sk', 'ws_ship_cdemo_sk', 'ws_ship_hdemo_sk', 'ws_ship_addr_sk', 'ws_web_page_sk', 'ws_web_site_sk', 'ws_ship_mode_sk', 'ws_warehouse_sk', 'ws_promo_sk', 'ws_order_number', 'ws_quantity', 'ws_wholesale_cost', 'ws_list_price', 'ws_sales_price', 'ws_ext_discount_amt', 'ws_ext_sales_price', 'ws_ext_wholesale_cost', 'ws_ext_list_price', 'ws_ext_tax', 'ws_coupon_amt', 'ws_ext_ship_cost', 'ws_net_paid', 'ws_net_paid_inc_tax', 'ws_net_paid_inc_ship', 'ws_net_paid_inc_ship_tax', 'ws_net_profit'],
        'customer_address': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'],
        'catalog_sales': ['cs_sold_date_sk', 'cs_sold_time_sk', 'cs_ship_date_sk', 'cs_bill_customer_sk', 'cs_bill_cdemo_sk', 'cs_bill_hdemo_sk', 'cs_bill_addr_sk', 'cs_ship_customer_sk', 'cs_ship_cdemo_sk', 'cs_ship_hdemo_sk', 'cs_ship_addr_sk', 'cs_call_center_sk', 'cs_catalog_page_sk', 'cs_ship_mode_sk', 'cs_warehouse_sk', 'cs_item_sk', 'cs_promo_sk', 'cs_order_number', 'cs_quantity', 'cs_wholesale_cost', 'cs_list_price', 'cs_sales_price', 'cs_ext_discount_amt', 'cs_ext_sales_price', 'cs_ext_wholesale_cost', 'cs_ext_list_price', 'cs_ext_tax', 'cs_coupon_amt', 'cs_ext_ship_cost', 'cs_net_paid', 'cs_net_paid_inc_tax', 'cs_net_paid_inc_ship', 'cs_net_paid_inc_ship_tax', 'cs_net_profit'],
        'reason': ['r_reason_sk', 'r_reason_id', 'r_reason_desc'],
        'catalog_returns': ['cr_returned_date_sk', 'cr_returned_time_sk', 'cr_item_sk', 'cr_refunded_customer_sk', 'cr_refunded_cdemo_sk', 'cr_refunded_hdemo_sk', 'cr_refunded_addr_sk', 'cr_returning_customer_sk', 'cr_returning_cdemo_sk', 'cr_returning_hdemo_sk', 'cr_returning_addr_sk', 'cr_call_center_sk', 'cr_catalog_page_sk', 'cr_ship_mode_sk', 'cr_warehouse_sk', 'cr_reason_sk', 'cr_order_number', 'cr_return_quantity', 'cr_return_amount', 'cr_return_tax', 'cr_return_amt_inc_tax', 'cr_fee', 'cr_return_ship_cost', 'cr_refunded_cash', 'cr_reversed_charge', 'cr_store_credit', 'cr_net_loss'],
        'time_dim': ['t_time_sk', 't_time_id', 't_time', 't_hour', 't_minute', 't_second', 't_am_pm', 't_shift', 't_sub_shift', 't_meal_time'],
        'ship_mode': ['sm_ship_mode_sk', 'sm_ship_mode_id', 'sm_type', 'sm_code', 'sm_carrier', 'sm_contract'],
        'income_band': ['ib_income_band_sk', 'ib_lower_bound', 'ib_upper_bound'],
        'store': ['s_store_sk', 's_store_id', 's_rec_start_date', 's_rec_end_date', 's_closed_date_sk', 's_store_name', 's_number_employees', 's_floor_space', 's_hours', 's_manager', 's_market_id', 's_geography_class', 's_market_desc', 's_market_manager', 's_division_id', 's_division_name', 's_company_id', 's_company_name', 's_street_number', 's_street_name', 's_street_type', 's_suite_number', 's_city', 's_county', 's_state', 's_zip', 's_country', 's_gmt_offset', 's_tax_precentage'],
        'inventory': ['inv_date_sk', 'inv_item_sk', 'inv_warehouse_sk', 'inv_quantity_on_hand'],
}


def load_dictionary(path):
    if path == None:
        return dict()
    word_vectors = KeyedVectors.load(path, mmap='r')
    return word_vectors

def load_numeric_min_max(path):
    with open(path,'r') as f:
        min_max_column = json.loads(f.read())
    return min_max_column

def load_numeric_min_max_csv(path):
    with open(path, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
        min_max_column = dict()
        for i, row in enumerate(data_raw):
            if i == 0:
                continue
            table_name, column_name = row[0].split(".")
            if table_name not in min_max_column:
                min_max_column[table_name] = dict()
            if column_name not in min_max_column[table_name]:
                min_max_column[table_name][column_name] = dict()
            min_max_column[table_name][column_name]['min'] = float(row[1])
            min_max_column[table_name][column_name]['max'] = float(row[2])
    return min_max_column


def determine_prefix(column, is_imdb = True):
    relation_name = column.split('.')[0]
    column_name = column.split('.')[1]
    if is_imdb:
        if relation_name == 'aka_title':
            if column_name == 'title':
                return 'title_'
            elif column_name == 'imdb_index':
                    return 'imdb_index_'
            elif column_name == 'phonetic_code':
                return 'phonetic_code_'
            elif column_name == 'note':
                return 'note_'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'char_name':
            if column_name == 'name':
                return 'name_'
            elif column_name == 'name_pcode_nf':
                return 'nf_'
            elif column_name == 'surname_pcode':
                return 'surname_'
            elif column_name == 'imdb_index':
                    return 'imdb_index_'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'movie_info_idx':
            if column_name == 'info':
                return 'info_'
            elif column_name == 'note':
                return 'note_'
            else:
                print (column)
                raise
        elif relation_name == 'title':
            if column_name == 'title':
                return 'title_'
            elif column_name == 'imdb_index':
                    return 'imdb_index_'
            elif column_name == 'phonetic_code':
                return 'phonetic_code_'
            elif column_name == 'series_years':
                return 'series_years'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'role_type':
            if column_name == 'role':
                return 'role_'
            else:
                print (column)
                raise
        elif relation_name == 'movie_companies':
            if column_name == 'note':
                return 'note_'
            else:
                print (column)
                raise
        elif relation_name == 'info_type':
            if column_name == 'info':
                return 'info_'
            else:
                print (column)
                raise
        elif relation_name == 'company_type':
            if column_name == 'kind':
                return ''
            else:
                print (column)
                raise
        elif relation_name == 'company_name':
            if column_name == 'name':
                return 'cn_name_'
            elif column_name == 'country_code':
                return 'country_'
            elif column_name == 'name_pcode_sf':
                    return 'sf_'
            elif column_name == 'name_pcode_nf':
                return 'nf_'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'keyword':
            if column_name == 'keyword':
                return 'keyword_'
            elif column_name == 'phonetic_code':
                return 'phonetic_code_'
            else:
                print (column)
                raise

        elif relation_name == 'movie_info':
            if column_name == 'info':
                return ''
            elif column_name == 'note':
                return 'note_'
            else:
                print (column)
                raise
        elif relation_name == 'name':
            if column_name == 'gender':
                return 'gender_'
            elif column_name == 'name':
                return 'name_'
            elif column_name == 'name_pcode_cf':
                return 'cf_'
            elif column_name == 'name_pcode_nf':
                return 'nf_'
            elif column_name == 'surname_pcode':
                return 'surname_'
            elif column_name == 'imdb_index':
                return 'imdb_index_'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'aka_name':
            if column_name == 'name':
                return 'name_'
            elif column_name == 'name_pcode_cf':
                return 'cf_'
            elif column_name == 'name_pcode_nf':
                return 'nf_'
            elif column_name == 'surname_pcode':
                return 'surname_'
            elif column_name == 'imdb_index':
                return 'imdb_index_'
            elif column_name == 'md5sum':
                return 'md5sum_'
            else:
                print (column)
                raise
        elif relation_name == 'link_type':
            if column_name == 'link':
                return 'link_'
            else:
                print (column)
                raise
        elif relation_name == 'person_info':
            if column_name == 'note':
                return 'note_'
            elif column_name == 'info':
                return ''
            else:
                print (column)
                raise
        elif relation_name == 'cast_info':
            if column_name == 'note':
                return 'note_'
            else:
                print (column)
                raise
        elif relation_name == 'comp_cast_type':
            if column_name == 'kind':
                return 'kind_'
            else:
                print (column)
                raise
        elif relation_name == 'kind_type':
            if column_name == 'kind':
                return 'kind_'
            else:
                print (column)
                raise
        else:
            print (relation_name)
            print (column)
            raise
    else: #tpcds
        # remove the table_name alias in tpcds column name
        prefix = column_name[column_name.find("_") + 1 : ]
        return prefix

def obtain_upper_bound_query_size(path):
    plan_node_max_num = 0
    condition_max_num = 0
    cost_label_max = 0.0
    cost_label_min = 9999999999.0
    card_label_max = 0.0
    card_label_min = 9999999999.0
    plans = []
    with open(path, 'r') as f:
        for plan in f.readlines():
            plan = json.loads(plan)
            plans.append(plan)
            cost = plan['cost']
            cardinality = plan['cardinality']
            if cost > cost_label_max:
                cost_label_max = cost
            elif cost < cost_label_min:
                cost_label_min = cost
            if cardinality > card_label_max:
                card_label_max = cardinality
            elif cardinality < card_label_min:
                card_label_min = cardinality
            sequence = plan['seq']
            plan_node_num = len(sequence)
            if plan_node_num > plan_node_max_num:
                plan_node_max_num = plan_node_num
            for node in sequence:
                if node == None:
                    continue
                if 'condition_filter' in node:
                    condition_num = len(node['condition_filter'])
                    if condition_num > condition_max_num:
                        condition_max_num = condition_num
                if 'condition_index' in node:
                    condition_num = len(node['condition_index'])
                    if condition_num > condition_max_num:
                        condition_max_num = condition_num
    cost_label_min, cost_label_max = math.log(cost_label_min), math.log(cost_label_max)
    card_label_min, card_label_max = math.log(card_label_min), math.log(card_label_max)
    print (plan_node_max_num, condition_max_num)
    print (cost_label_min, cost_label_max)
    print (card_label_min, card_label_max)
    return plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max

def obtain_upper_bound_query_size_nocost(path):
    plan_node_max_num = 0
    condition_max_num = 0
    card_label_max = 0.0
    card_label_min = 9999999999.0
    plans = []
    with open(path, 'r') as f:
        for plan in f.readlines():
            plan = json.loads(plan)
            plans.append(plan)
            cardinality = plan['cardinality']
            if cardinality > card_label_max:
                card_label_max = cardinality
            elif cardinality < card_label_min:
                card_label_min = cardinality
            sequence = plan['seq']
            plan_node_num = len(sequence)
            if plan_node_num > plan_node_max_num:
                plan_node_max_num = plan_node_num
            for node in sequence:
                if node == None:
                    continue
                if 'condition_filter' in node:
                    condition_num = len(node['condition_filter'])
                    if condition_num > condition_max_num:
                        condition_max_num = condition_num
                if 'condition_index' in node:
                    condition_num = len(node['condition_index'])
                    if condition_num > condition_max_num:
                        condition_max_num = condition_num
    print (card_label_min, card_label_max)
    card_label_min, card_label_max = math.log(card_label_min), math.log(card_label_max)
    print (plan_node_max_num, condition_max_num)
    print (card_label_min, card_label_max)
    return plan_node_max_num, condition_max_num, card_label_min, card_label_max


def prepare_dataset(database):

    column2pos = dict()

    tables = ['aka_name', 'aka_title', 'cast_info', 'char_name', 'company_name', 'company_type', 'comp_cast_type', 'complete_cast', 'info_type', 'keyword', 'kind_type', 'link_type', 'movie_companies', 'movie_info', 'movie_info_idx',
              'movie_keyword', 'movie_link', 'name', 'person_info', 'role_type', 'title']

    for table_name in tables:
        column2pos[table_name] = database[table_name].columns

    indexes = ['aka_name_pkey', 'aka_title_pkey', 'cast_info_pkey', 'char_name_pkey',
               'comp_cast_type_pkey', 'company_name_pkey', 'company_type_pkey', 'complete_cast_pkey',
               'info_type_pkey', 'keyword_pkey', 'kind_type_pkey', 'link_type_pkey', 'movie_companies_pkey',
               'movie_info_idx_pkey', 'movie_keyword_pkey', 'movie_link_pkey', 'name_pkey', 'role_type_pkey',
               'title_pkey', 'movie_info_pkey', 'person_info_pkey', 'company_id_movie_companies',
               'company_type_id_movie_companies', 'info_type_id_movie_info_idx', 'info_type_id_movie_info',
               'info_type_id_person_info', 'keyword_id_movie_keyword', 'kind_id_aka_title', 'kind_id_title',
               'linked_movie_id_movie_link', 'link_type_id_movie_link', 'movie_id_aka_title', 'movie_id_cast_info',
               'movie_id_complete_cast', 'movie_id_movie_ companies', 'movie_id_movie_info_idx',
               'movie_id_movie_keyword', 'movie_id_movie_link', 'movie_id_movie_info', 'person_id_aka_name',
               'person_id_cast_info', 'person_id_person_info', 'person_role_id_cast_info', 'role_id_cast_info']
    indexes_id = dict()
    for idx, index in enumerate(indexes):
        indexes_id[index] = idx + 1
    physic_ops_id = {'Materialize':1, 'Sort':2, 'Hash':3, 'Merge Join':4, 'Bitmap Index Scan':5,
                     'Index Only Scan':6, 'BitmapAnd':7, 'Nested Loop':8, 'Aggregate':9, 'Result':10,
                     'Hash Join':11, 'Seq Scan':12, 'Bitmap Heap Scan':13, 'Index Scan':14, 'BitmapOr':15}
    strategy_id = {'Plain':1}
    compare_ops_id = {'=':1, '>':2, '<':3, '!=':4, '~~':5, '!~~':6, '!Null': 7, '>=':8, '<=':9}
    bool_ops_id = {'AND':1,'OR':2}
    tables_id = {}
    columns_id = {}
    table_id = 1
    column_id = 1
    for table_name in tables:
        tables_id[table_name] = table_id
        table_id += 1
        for column in column2pos[table_name]:
            columns_id[table_name+'.'+column] = column_id
            column_id += 1
    return column2pos, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, tables

def prepare_dataset_general(database, dbname, columns = None):

    column2pos = dict()

    tables = [t for t in database]

    for table_name in tables:
        column2pos[table_name] = database[table_name].columns

    if dbname == "imdb-small":
        indexes = IMDB_SMALL_INDEXES
    elif dbname == "imdb-medium":
        indexes = IMDB_MEDIUM_INDEXES
    elif dbname == "imdb-large":
        indexes = IMDB_LARGE_INDEXES
    # elif dbname == "tpcds-bench":
    #     indexes = TPCDS_BENCH_INDEXES
    # elif dbname == "tpcds-large":
    #     indexes = TPCDS_LARGE_INDEXES
    elif "syn-multi" in dbname:
        indexes = SYN_M_INDEXES
    elif "syn-single" in dbname:
        indexes = SYN_S_INDEXES
    
    indexes_id = dict()
    for idx, index in enumerate(indexes):
        indexes_id[index] = idx + 1
    physic_ops_id = {'Materialize':1, 'Sort':2, 'Hash':3, 'Merge Join':4, 'Bitmap Index Scan':5,
                     'Index Only Scan':6, 'BitmapAnd':7, 'Nested Loop':8, 'Aggregate':9, 'Result':10,
                     'Hash Join':11, 'Seq Scan':12, 'Bitmap Heap Scan':13, 'Index Scan':14, 'BitmapOr':15}
    strategy_id = {'Plain':1}
    # compare_ops_id = {'=':1, '>':2, '<':3, '!=':4, '~~':5, '!~~':6, '!Null': 7, '>=':8, '<=':9}
    compare_ops_id = {'=':1, '>':2, '<':3, '!=':4, '~~':5, '!~~':6, '!Null': 7, '>=':8, '<=':9, 'IS':10}
    bool_ops_id = {'AND':1,'OR':2}
    tables_id = {}
    columns_id = {}
    table_id = 1
    column_id = 1
    if columns == None:
        for table_name in tables:
            tables_id[table_name] = table_id
            table_id += 1
            for column in column2pos[table_name]:
                columns_id[table_name+'.'+column] = column_id
                column_id += 1
    else:
        for table_name in tables:
            if table_name not in columns:
                continue
            tables_id[table_name] = table_id
            table_id += 1
            for column in column2pos[table_name]:
                if column in columns[table_name]:
                    columns_id[table_name+'.'+column] = column_id
                    column_id += 1
    return column2pos, indexes_id, tables_id, columns_id, physic_ops_id, compare_ops_id, bool_ops_id, tables

def prepare_loading(dbname):
    if "imdb" in dbname:
        minmax_path = "minmax/imdb.csv"
        wordvector_path = f"../wordvectors/{dbname}/wordvectors_updated.kv"
        if dbname == "imdb-small": wordvector_path = None
    # elif "tpcds" in dbname:
    #     minmax_path = "../minmax/tpcds.csv"
    #     wordvector_path = f"../wordvectors/{dbname}/wordvectors_updated.kv"
    elif "syn-multi" in dbname:
        id = dbname[-2:]
        minmax_path = f"minmax/synthetic/multi/minmax_{id}.csv"
        wordvector_path = None
    elif "syn-single" in dbname:
        id = dbname[-2:]
        minmax_path = f"minmax/synthetic/single/minmax_{id}.csv"
        wordvector_path = None

    if dbname == "imdb-small":
        columns = IMDB_SMALL_COLS
    elif dbname == "imdb-medium":
        columns = IMDB_MEDIUM_COLS
    elif dbname == "imdb-large":
        columns = IMDB_LARGE_COLS
    # elif dbname == "tpcds-bench":
    #     columns = TPCDS_BENCH_COLS
    # elif dbname == "tpcds-large":
    #     columns = TPCDS_LARGE_COLS
    else:
        columns = None


    return minmax_path, wordvector_path, columns
