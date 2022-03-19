"""Registry of datasets and schemas."""
import collections
import os
import pickle

import numpy as np
import pandas as pd

import collections
from common import CsvTable

dataset_list = ['imdb', 'tpcds','toy','synthetic']

def CachedReadCsv(filepath, **kwargs):
    """Wrapper around pd.read_csv(); accepts same arguments."""
    parsed_path = filepath[:-4] + '.df'
    if os.path.exists(parsed_path):
        with open(parsed_path, 'rb') as f:
            df = pickle.load(f)
        assert isinstance(df, pd.DataFrame), type(df)
        print('Loaded parsed csv from', parsed_path)
    else:
        df = pd.read_csv(filepath, **kwargs)
        with open(parsed_path, 'wb') as f:
            # Use protocol=4 since we expect df >= 4GB.
            pickle.dump(df, f, protocol=4)
        print('Saved parsed csv to', parsed_path)
    return df


class JoinOrderBenchmark(object):
    ALIAS_TO_TABLE_NAME = {
        'ci': 'cast_info',
        'ct': 'company_type',
        'mc': 'movie_companies',
        't': 'title',
        'cn': 'company_name',
        'k': 'keyword',
        'mi_idx': 'movie_info_idx',
        'it': 'info_type',
        'mi': 'movie_info',
        'mk': 'movie_keyword',
    }

    # Columns where only equality filters make sense.
    CATEGORICAL_COLUMNS = collections.defaultdict(
        list,
        {
            # 216
            'company_name': ['country_code'],
            'keyword': [],
            # 113
            'info_type': ['info'],
            # 4
            'company_type': ['kind'],
            # 2
            'movie_companies': ['company_type_id'],
            # Dom size 134K.
            'movie_keyword': ['keyword_id'],
            # 7
            'title': ['kind_id'],
            # 5
            'movie_info_idx': ['info_type_id'],
            # 11
            'cast_info': ['role_id'],
            # 71
            'movie_info': ['info_type_id'],
        })

    # Columns with a reasonable range/IN interpretation.
    RANGE_COLUMNS = collections.defaultdict(
        list,
        {
            # 18487, 17447
            'company_name': ['name_pcode_cf', 'name_pcode_sf'],
            # 15482
            'keyword': ['phonetic_code'],
            'info_type': [],
            'company_type': [],
            'movie_companies': [],
            'movie_keyword': [],
            # 26, 133, 23260, 97, 14907, 1409
            'title': [
                'imdb_index', 'production_year', 'phonetic_code', 'season_nr',
                'episode_nr', 'series_years'
            ],
            'movie_info_idx': [],
            # 1095
            'cast_info': ['nr_order'],
            'movie_info': [],
        })

    CSV_FILES = [
        'name.csv', 'movie_companies.csv', 'aka_name.csv', 'movie_info.csv',
        'movie_keyword.csv', 'person_info.csv', 'comp_cast_type.csv',
        'complete_cast.csv', 'char_name.csv', 'movie_link.csv',
        'company_type.csv', 'cast_info.csv', 'info_type.csv',
        'company_name.csv', 'aka_title.csv', 'kind_type.csv', 'role_type.csv',
        'movie_info_idx.csv', 'keyword.csv', 'link_type.csv', 'title.csv'
    ]
    UAE_PRED_COLS = collections.defaultdict(list,{
           'movie_info_idx.csv' : ['movie_id','info_type_id'],
            'company_name.csv' : ['country_code', 'name_pcode_sf', 'name_pcode_cf'],
            'cast_info.csv' : ['movie_id','person_id', 'role_id', 'nr_order'],
            'company_type.csv' : ['kind'],
            'movie_keyword.csv' : ['movie_id','keyword_id'],
            'movie_companies.csv' : ['movie_id','company_id', 'company_type_id'],
            'info_type.csv' : ['info'],
            'keyword.csv' : ['phonetic_code'],
            'title.csv' : ['id','production_year', 'phonetic_code', 'imdb_index', 'episode_nr', 'series_years', 'kind_id', 'season_nr'],
            'movie_info.csv' : ['movie_id','info_type_id'],

        })
    BASE_TABLE_PRED_COLS = collections.defaultdict(
        list,
        {
            'movie_info_idx.csv': ['info_type_id', 'movie_id'],
            'movie_companies.csv': [
                'company_id', 'company_type_id', 'movie_id'
            ],
            # Column(kind_id, distribution_size=7), Column(production_year,
            # distribution_size=133),
            'title.csv': ['id', 'kind_id', 'production_year'],
            'cast_info.csv': ['movie_id', 'role_id'],
            # Column(keyword_id, distribution_size=134170)
            'movie_keyword.csv': ['movie_id', 'keyword_id'],
            # Column(info_type_id, distribution_size=71), Column(info,
            # distribution_size=2720930), Column(note, distribution_size=133604
            'movie_info.csv': [
                'movie_id',
                'info_type_id',
            ],
            'comp_cast_type.csv': ['id', 'kind'],
            'aka_name.csv': ['id', 'person_id'],
            'name.csv': ['id'],
        })
    JOB_UNION_PRED_COLS = collections.defaultdict(
        list, {
            'title.csv': [
                'id', 'kind_id', 'title', 'production_year', 'episode_nr'
            ],
            'aka_title.csv': ['movie_id'],
            'cast_info.csv': ['movie_id', 'note','role_id'],
            'complete_cast.csv': ['subject_id', 'movie_id'],
            'movie_companies.csv': [
                'company_id', 'company_type_id', 'movie_id', 'note'
            ],
            'movie_info.csv': ['movie_id','info_type_id', 'info', 'note'],
            'movie_info_idx.csv': ['info_type_id', 'movie_id', 'info'],
            'movie_keyword.csv': ['keyword_id', 'movie_id'],
            'movie_link.csv': ['link_type_id', 'movie_id'],
            'kind_type.csv': ['id', 'kind'],
            'comp_cast_type.csv': ['id', 'kind'],
            'company_name.csv': ['id', 'country_code', 'name'],
            'company_type.csv': ['id', 'kind'],
            'info_type.csv': ['id', 'info'],
            'keyword.csv': ['id', 'keyword'],
            'link_type.csv': ['id', 'link'],
        })
    JOB_TOY_PRED_COLS = collections.defaultdict(
        list,{
            'title.csv': ['id', 'kind_id', 'production_year'],
            'movie_keyword.csv': ['movie_id', 'keyword_id'],
            'movie_companies.csv': [
                'company_id', 'company_type_id', 'movie_id'
            ],
        }
    )

    JOB_M_ADD_KEY_PRED_COLS = collections.defaultdict(
        list, {

                'title.csv': ['id', 'kind_id', 'title', 'production_year','imdb_index','phonetic_code', 'season_nr','episode_nr', 'series_years'],
                 'aka_title.csv': ['movie_id'],
                 'cast_info.csv': ['movie_id', 'note', 'role_id','nr_order'],
                 'complete_cast.csv': ['subject_id', 'movie_id'],
                 'movie_companies.csv': ['company_id', 'company_type_id', 'movie_id', 'note'],
                 'movie_info.csv': ['movie_id', 'info', 'note', 'info_type_id'],
                 'movie_info_idx.csv': ['info_type_id', 'movie_id', 'info'],
                 'movie_keyword.csv': ['keyword_id', 'movie_id'],
                 'movie_link.csv': ['link_type_id', 'movie_id'],
                 'kind_type.csv': ['id', 'kind'],
                 'comp_cast_type.csv': ['id', 'kind'],
                 'company_name.csv': ['id', 'country_code', 'name','name_pcode_nf', 'name_pcode_sf'],
                 'company_type.csv': ['id', 'kind'],
                 'info_type.csv': ['id', 'info'],
                 'keyword.csv': ['id', 'keyword','phonetic_code'],
                 'link_type.csv': ['id', 'link']
        })
    JOB_ALL_PRED_COLS = collections.defaultdict(
        list, {
                'title.csv': ['id', 'kind_id', 'title', 'production_year', 'episode_nr'],
                 'aka_title.csv': ['movie_id'],
                 'cast_info.csv': ['movie_id', 'note', 'role_id','person_role_id','person_id'],
                 'complete_cast.csv': ['subject_id', 'movie_id'],
                 'movie_companies.csv': ['company_id', 'company_type_id', 'movie_id', 'note'],
                 'movie_info.csv': ['movie_id', 'info', 'note', 'info_type_id'],
                 'movie_info_idx.csv': ['info_type_id', 'movie_id', 'info'],
                 'movie_keyword.csv': ['keyword_id', 'movie_id'],
                 'movie_link.csv': ['link_type_id', 'movie_id'],
                 'kind_type.csv': ['id', 'kind'],
                 'comp_cast_type.csv': ['id', 'kind'],
                 'company_name.csv': ['id', 'country_code', 'name'],
                 'company_type.csv': ['id', 'kind'],
                 'info_type.csv': ['id', 'info'],
                 'keyword.csv': ['id', 'keyword'],
                 'link_type.csv': ['id', 'link'],

                'char_name.csv': ['id','name'],
                'role_type.csv': ['id','role'],
                'aka_name.csv': ['person_id','name'],
                'name.csv': ['id','name','gender'],
                'person_info.csv': ['info_type_id','note'],

        })
    JOB_M_PRED_COLS = collections.defaultdict(
        list, {
            'title.csv': [
                'id', 'kind_id', 'title', 'production_year', 'episode_nr'
            ],
            'aka_title.csv': ['movie_id'],
            'cast_info.csv': ['movie_id', 'note'],
            'complete_cast.csv': ['subject_id', 'movie_id'],
            'movie_companies.csv': [
                'company_id', 'company_type_id', 'movie_id', 'note'
            ],
            'movie_info.csv': ['movie_id', 'info', 'note'],
            'movie_info_idx.csv': ['info_type_id', 'movie_id', 'info'],
            'movie_keyword.csv': ['keyword_id', 'movie_id'],
            'movie_link.csv': ['link_type_id', 'movie_id'],
            'kind_type.csv': ['id', 'kind'],
            'comp_cast_type.csv': ['id', 'kind'],
            'company_name.csv': ['id', 'country_code', 'name'],
            'company_type.csv': ['id', 'kind'],
            'info_type.csv': ['id', 'info'],
            'keyword.csv': ['id', 'keyword'],
            'link_type.csv': ['id', 'link'],
        })

    JOB_FULL_PRED_COLS = collections.defaultdict(
        list, {
            'title.csv': [
                'id', 'kind_id', 'title', 'production_year', 'episode_nr'
            ],
            'aka_name.csv': ['person_id'],
            'aka_title.csv': ['movie_id'],
            'cast_info.csv': [
                'person_id', 'person_role_id', 'role_id', 'movie_id', 'note'
            ],
            'char_name.csv': ['id'],
            'comp_cast_type.csv': ['id', 'kind'],
            'comp_cast_type__complete_cast__status_id.csv': ['id', 'kind'],
            'comp_cast_type__complete_cast__subject_id.csv': ['id', 'kind'],
            'company_name.csv': ['id', 'country_code', 'name'],
            'company_type.csv': ['id', 'kind'],
            'complete_cast': ['status_id', 'subject_id', 'movie_id'],
            'info_type.csv': ['id', 'info'],
            'info_type__movie_info__info_type_id.csv': ['id', 'info'],
            'info_type__movie_info_idx__info_type_id.csv': ['id', 'info'],
            'info_type__person_info__info_type_id.csv': ['id', 'info'],
            'keyword.csv': ['id', 'keyword'],
            'kind_type.csv': ['id', 'kind'],
            'link_type.csv': ['id', 'link'],
            'movie_companies.csv': [
                'company_id', 'company_type_id', 'movie_id', 'note'
            ],
            'movie_info_idx.csv': ['info_type_id', 'movie_id', 'info'],
            'movie_info.csv': ['info_type_id', 'movie_id', 'info', 'note'],
            'movie_keyword.csv': ['keyword_id', 'movie_id'],
            'movie_link.csv': ['link_type_id', 'movie_id', 'linked_movie_id'],
            'name.csv': ['id'],
            'person_info.csv': ['person_id', 'info_type_id'],
            'role_type.csv': ['id'],
        })

    IMDB_DB_PRED_COLS = collections.defaultdict(
        list, {
            'name.csv': ['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'] ,
            'movie_companies.csv': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'] ,
            'aka_name.csv': ['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'] ,
            'movie_info.csv': ['id', 'movie_id', 'info_type_id', 'info', 'note'] ,
            'movie_keyword.csv': ['id', 'movie_id', 'keyword_id'] ,
            'person_info.csv': ['id', 'person_id', 'info_type_id', 'info', 'note'] ,
            'comp_cast_type.csv': ['id', 'kind'] ,
            'complete_cast.csv': ['id', 'movie_id', 'subject_id', 'status_id'] ,
            'char_name.csv': ['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf', 'surname_pcode', 'md5sum'] ,
            'movie_link.csv': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'] ,
            'company_type.csv': ['id', 'kind'] ,
            'cast_info.csv': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'] ,
            'info_type.csv': ['id', 'info'] ,
            'company_name.csv': ['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'] ,
            'aka_title.csv': ['id', 'movie_id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'note', 'md5sum'] ,
            'kind_type.csv': ['id', 'kind'] ,
            'role_type.csv': ['id', 'role'] ,
            'movie_info_idx.csv': ['id', 'movie_id', 'info_type_id', 'info', 'note'] ,
            'keyword.csv': ['id', 'keyword', 'phonetic_code'] ,
            'link_type.csv': ['id', 'link'] ,
            'title.csv': ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'] ,
        })

    IMDB_DEEPDB_PRED_COLS = collections.defaultdict(
        list, {'name.csv' : ['surname_pcode', 'imdb_index', 'gender', 'id'],
                'movie_companies.csv' : ['movie_id', 'company_type_id', 'company_id', 'id'],
                'aka_name.csv' : ['surname_pcode', 'imdb_index', 'person_id', 'id'],
                'movie_info.csv' : ['movie_id', 'id', 'info_type_id'],
                'movie_keyword.csv' : ['movie_id', 'id', 'keyword_id'],
                'person_info.csv' : ['person_id', 'id', 'info_type_id'],
                'comp_cast_type.csv' : ['kind', 'id'],
                'complete_cast.csv' : ['movie_id', 'subject_id', 'status_id', 'id'],
                'char_name.csv' : ['imdb_index', 'id'],
                'movie_link.csv' : ['movie_id', 'link_type_id', 'linked_movie_id', 'id'],
                'company_type.csv' : ['kind', 'id'],
                'cast_info.csv' : ['nr_order', 'person_role_id', 'id', 'person_id', 'movie_id', 'role_id'],
                'info_type.csv' : ['info', 'id'],
                'company_name.csv' : ['country_code', 'id'],
                'aka_title.csv' : ['episode_nr', 'id', 'imdb_index', 'season_nr', 'episode_of_id', 'production_year', 'note', 'movie_id', 'kind_id'],
                'kind_type.csv' : ['kind', 'id'],
                'role_type.csv' : ['role', 'id'],
                'movie_info_idx.csv' : ['movie_id', 'id', 'info_type_id'],
                'keyword.csv' : ['id'],
                'link_type.csv' : ['link', 'id'],
                'title.csv' : ['series_years', 'episode_nr', 'id', 'imdb_index', 'season_nr', 'episode_of_id', 'production_year', 'kind_id']})


    JOB_DEEPDB_PRED_COLS = collections.defaultdict(
        list, {
                'title.csv' : ['episode_nr', 'kind_id', 'production_year', 'id'],
                'movie_info_idx.csv' : ['movie_id', 'info_type_id'],
                'movie_info.csv' : ['movie_id', 'info_type_id'],
                'info_type.csv' : ['info', 'id'],
                'complete_cast.csv' : ['subject_id', 'movie_id'],
                'cast_info.csv' : ['movie_id', 'role_id'],
                'comp_cast_type.csv' : ['kind', 'id'],
                'movie_keyword.csv' : ['movie_id', 'keyword_id'],
                'keyword.csv' : ['id'],
                'movie_companies.csv' : ['movie_id', 'company_type_id', 'company_id'],
                'company_name.csv' : ['country_code', 'id'],
                'company_type.csv' : ['kind', 'id'],
                'aka_title.csv' : ['movie_id'],
                'kind_type.csv' : ['kind', 'id'],
                'movie_link.csv' : ['movie_id', 'link_type_id'],
                'link_type.csv' : ['link', 'id'],
        })

    JOB_ORIGINAL_PRED_COLS = collections.defaultdict(list, {
        'cast_info.csv': ['movie_id', 'note', 'person_role_id', 'role_id', 'person_id'],
         'char_name.csv': ['id', 'name'],
         'company_name.csv': ['country_code', 'id', 'name'],
         'company_type.csv': ['id', 'kind'],
         'movie_companies.csv': ['company_id', 'company_type_id', 'movie_id', 'note'],
         'role_type.csv': ['id', 'role'],
         'title.csv': ['id', 'production_year', 'title', 'kind_id', 'episode_nr'],
         'keyword.csv': ['id', 'keyword'],
         'link_type.csv': ['id', 'link'],
         'movie_keyword.csv': ['keyword_id', 'movie_id'],
         'movie_link.csv': ['link_type_id', 'movie_id', 'linked_movie_id'],
         'info_type.csv': ['id', 'info'],
         'movie_info.csv': ['info', 'info_type_id', 'movie_id', 'note'],
         'movie_info_idx.csv': ['info', 'info_type_id', 'movie_id'],
         'kind_type.csv': ['id', 'kind'],
         'aka_title.csv': ['movie_id', 'title'],
         'aka_name.csv': ['name', 'person_id'],
         'name.csv': ['id', 'name', 'gender', 'name_pcode_cf'],
         'comp_cast_type.csv': ['id', 'kind'],
         'complete_cast.csv': ['movie_id', 'status_id', 'subject_id'],
         'person_info.csv': ['info_type_id', 'person_id', 'note', 'info']})

    JOB_ORIGINAL_DUP_PRED = collections.defaultdict(list,{
        'cast_info.csv': ['movie_id', 'note', 'person_role_id', 'role_id', 'person_id'], 'char_name.csv': ['id', 'name'], 'company_name.csv': ['country_code', 'id', 'name'], 'company_type.csv': ['id', 'kind'], 'movie_companies.csv': ['company_id', 'company_type_id', 'movie_id', 'note'], 'role_type.csv': ['id', 'role'], 'title.csv': ['id', 'production_year', 'title', 'kind_id', 'episode_nr'], 'keyword.csv': ['id', 'keyword'], 'link_type.csv': ['id', 'link'], 'movie_keyword.csv': ['keyword_id', 'movie_id'], 'movie_link.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'info_type.csv': ['id', 'info'], 'movie_info.csv': ['info', 'info_type_id', 'movie_id', 'note'], 'movie_info_idx.csv': ['info', 'info_type_id', 'movie_id'], 'kind_type.csv': ['id', 'kind'], 'aka_title.csv': ['movie_id', 'title'], 'aka_name.csv': ['name', 'person_id'], 'name.csv': ['id', 'name', 'gender', 'name_pcode_cf'], 'comp_cast_type.csv': ['id', 'kind'], 'complete_cast.csv': ['movie_id', 'status_id', 'subject_id'], 'person_info.csv': ['info_type_id', 'person_id', 'note', 'info'], 'aka_name_dup_1_.csv': ['name', 'person_id'], 'cast_info_dup_1_.csv': ['movie_id', 'note', 'person_role_id', 'role_id', 'person_id'], 'cast_info_dup_2_.csv': ['movie_id', 'note', 'person_role_id', 'role_id', 'person_id'], 'movie_info_dup_1_.csv': ['info', 'info_type_id', 'movie_id', 'note'], 'person_info_dup_1_.csv': ['info_type_id', 'person_id', 'note', 'info'], 'info_type_dup_1_.csv': ['id', 'info'], 'person_info_dup_2_.csv': ['info_type_id', 'person_id', 'note', 'info'], 'person_info_dup_3_.csv': ['info_type_id', 'person_id', 'note', 'info'], 'movie_companies_dup_1_.csv': ['company_id', 'company_type_id', 'movie_id', 'note'], 'movie_keyword_dup_1_.csv': ['keyword_id', 'movie_id'], 'movie_info_idx_dup_1_.csv': ['info', 'info_type_id', 'movie_id'], 'movie_info_dup_2_.csv': ['info', 'info_type_id', 'movie_id', 'note'], 'cast_info_dup_3_.csv': ['movie_id', 'note', 'person_role_id', 'role_id', 'person_id'], 'movie_link_dup_1_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'movie_link_dup_2_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'aka_title_dup_1_.csv': ['movie_id', 'title'], 'movie_keyword_dup_2_.csv': ['keyword_id', 'movie_id'], 'movie_info_idx_dup_2_.csv': ['info', 'info_type_id', 'movie_id'], 'movie_info_dup_3_.csv': ['info', 'info_type_id', 'movie_id', 'note'], 'cast_info_dup_4_.csv': ['movie_id', 'note', 'person_role_id', 'role_id', 'person_id'], 'movie_link_dup_3_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'movie_link_dup_4_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'aka_title_dup_2_.csv': ['movie_id', 'title'], 'movie_info_idx_dup_3_.csv': ['info', 'info_type_id', 'movie_id'], 'movie_info_dup_4_.csv': ['info', 'info_type_id', 'movie_id', 'note'], 'cast_info_dup_5_.csv': ['movie_id', 'note', 'person_role_id', 'role_id', 'person_id'], 'movie_link_dup_5_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'movie_link_dup_6_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'aka_title_dup_3_.csv': ['movie_id', 'title'], 'movie_info_dup_5_.csv': ['info', 'info_type_id', 'movie_id', 'note'], 'cast_info_dup_6_.csv': ['movie_id', 'note', 'person_role_id', 'role_id', 'person_id'], 'movie_link_dup_7_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'movie_link_dup_8_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'aka_title_dup_4_.csv': ['movie_id', 'title'], 'cast_info_dup_7_.csv': ['movie_id', 'note', 'person_role_id', 'role_id', 'person_id'], 'movie_link_dup_9_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'movie_link_dup_10_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'aka_title_dup_5_.csv': ['movie_id', 'title'], 'movie_link_dup_11_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'movie_link_dup_12_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'aka_title_dup_6_.csv': ['movie_id', 'title'], 'movie_link_dup_13_.csv': ['link_type_id', 'movie_id', 'linked_movie_id'], 'title_dup_1_.csv': ['id', 'production_year', 'title', 'kind_id', 'episode_nr'], 'aka_title_dup_7_.csv': ['movie_id', 'title'], 'aka_title_dup_8_.csv': ['movie_id', 'title'], 'complete_cast_dup_1_.csv': ['movie_id', 'status_id', 'subject_id'], 'comp_cast_type_dup_1_.csv': ['id', 'kind']})

    IMDB_FULL_DUP_PRED_COLS = collections.defaultdict(list,{'name.csv': ['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], 'movie_companies.csv': ['id', 'movie_id', 'company_id', 'company_type_id', 'note'], 'aka_name.csv': ['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'], 'movie_info.csv': ['id', 'movie_id', 'info_type_id', 'info', 'note'], 'movie_keyword.csv': ['id', 'movie_id', 'keyword_id'], 'person_info.csv': ['id', 'person_id', 'info_type_id', 'info', 'note'], 'comp_cast_type.csv': ['id', 'kind'], 'complete_cast.csv': ['id', 'movie_id', 'subject_id', 'status_id'], 'char_name.csv': ['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf', 'surname_pcode', 'md5sum'], 'movie_link.csv': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'], 'company_type.csv': ['id', 'kind'], 'cast_info.csv': ['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'], 'info_type.csv': ['id', 'info'], 'company_name.csv': ['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'], 'aka_title.csv': ['id', 'movie_id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'note', 'md5sum'], 'kind_type.csv': ['id', 'kind'], 'role_type.csv': ['id', 'role'], 'movie_info_idx.csv': ['id', 'movie_id', 'info_type_id', 'info', 'note'], 'keyword.csv': ['id', 'keyword', 'phonetic_code'], 'link_type.csv': ['id', 'link'], 'title.csv': ['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'], 'person_info_dup_.csv': ['id', 'person_id', 'info_type_id', 'info', 'note'], 'comp_cast_type_dup_.csv': ['id', 'kind'], 'movie_link_dup_.csv': ['id', 'movie_id', 'linked_movie_id', 'link_type_id'], 'info_type_dup_.csv': ['id', 'info']})
    # For JOB-light schema.
    TRUE_FULL_OUTER_CARDINALITY = {

        ('movie_companies', 'movie_keyword', 'title'):38028991,
        ('cast_info', 'movie_keyword', 'title'): 241319266,
        ('cast_info', 'movie_companies', 'movie_info',\
         'movie_info_idx', 'movie_keyword', 'title'): 2128877229383,
        ('aka_title', 'cast_info', 'comp_cast_type', 'company_name',\
         'company_type', 'complete_cast', 'info_type', 'keyword', \
         'kind_type', 'link_type', 'movie_companies', 'movie_info', \
         'movie_info_idx', 'movie_keyword', 'movie_link', 'title'): 11244784701309,
        
    }

    # CSV -> RANGE union CATEGORICAL columns.
    _CONTENT_COLS = None

    @staticmethod
    def ContentColumns():
        if JoinOrderBenchmark._CONTENT_COLS is None:
            JoinOrderBenchmark._CONTENT_COLS = {
                '{}.csv'.format(table_name):
                range_cols + JoinOrderBenchmark.CATEGORICAL_COLUMNS[table_name]
                for table_name, range_cols in
                JoinOrderBenchmark.RANGE_COLUMNS.items()
            }
            # Add join keys.
            for table_name in JoinOrderBenchmark._CONTENT_COLS:
                cols = JoinOrderBenchmark._CONTENT_COLS[table_name]
                if table_name == 'title.csv':
                    cols.append('id')
                elif 'movie_id' in JoinOrderBenchmark.BASE_TABLE_PRED_COLS[
                        table_name]:
                    cols.append('movie_id')

        return JoinOrderBenchmark._CONTENT_COLS

    @staticmethod
    def GetFullOuterCardinalityOrFail(join_tables):
        key = tuple(sorted(join_tables))
        return JoinOrderBenchmark.TRUE_FULL_OUTER_CARDINALITY[key]

    @staticmethod
    def GetJobLightJoinKeys():
        return {
            'title': 'id',
            'cast_info': 'movie_id',
            'movie_companies': 'movie_id',
            'movie_info': 'movie_id',
            'movie_info_idx': 'movie_id',
            'movie_keyword': 'movie_id',
        }


def LoadImdb(table=None,
             data_dir='./datasets/job_csv_export/',
             try_load_parsed=True,
             use_cols='simple'):
    """Loads IMDB tables with a specified set of columns.

    use_cols:
      simple: only movie_id join keys (JOB-light)
      content: + content columns (JOB-light-ranges)
      multi: all join keys in JOB-M
      full: all join keys in JOB-full
      None: load all columns

    Returns:
      A single CsvTable if 'table' is specified, else a dict of CsvTables.
    """
    # assert use_cols in ['toy','simple', 'content', 'multi','imdb-db','imdb-full-dup','union','deepdb','imdb-full-deepdb', 'job','uae-light','original_dup',None], use_cols

    def TryLoad(table_name, filepath, use_cols, **kwargs):
        """Try load from previously parsed (table, columns)."""
        if use_cols:
            cols_str = '-'.join(use_cols)
            parsed_path = filepath[:-4] + '.{}.table'.format(cols_str)
        else:
            parsed_path = filepath[:-4] + '.table'
        if try_load_parsed:
            if os.path.exists(parsed_path):
                arr = np.load(parsed_path, allow_pickle=True)
                print('Loaded parsed Table from', parsed_path)
                table = arr.item()
                print(table)
                return table
        table = CsvTable(
            table_name,
            filepath,
            cols=use_cols,
            **kwargs,
        )
        if try_load_parsed:
            np.save(open(parsed_path, 'wb'), table)
            print('Saved parsed Table to', parsed_path)
        return table

    def get_use_cols(filepath):
        if use_cols == 'simple':
            return JoinOrderBenchmark.BASE_TABLE_PRED_COLS.get(filepath, None)
        elif use_cols == 'content':
            return JoinOrderBenchmark.ContentColumns().get(filepath, None)
        elif use_cols == 'multi':
            return JoinOrderBenchmark.JOB_M_PRED_COLS.get(filepath, None)
        elif use_cols == 'toy' :
            return JoinOrderBenchmark.JOB_TOY_PRED_COLS.get(filepath,None)
        elif use_cols =='imdb-db':
            return JoinOrderBenchmark.IMDB_DB_PRED_COLS.get(filepath,None)
        elif use_cols =='imdb-full-dup':
            return JoinOrderBenchmark.IMDB_FULL_DUP_PRED_COLS.get(filepath,None)
        elif use_cols == 'union':
            return JoinOrderBenchmark.JOB_UNION_PRED_COLS.get(filepath,None)
        elif use_cols == 'deepdb':
            return JoinOrderBenchmark.JOB_DEEPDB_PRED_COLS.get(filepath,None)
        elif use_cols == 'imdb-full-deepdb':
            return JoinOrderBenchmark.IMDB_DEEPDB_PRED_COLS.get(filepath,None)
        elif use_cols == 'job':
            return JoinOrderBenchmark.JOB_ORIGINAL_PRED_COLS.get(filepath,None)
        elif use_cols =='original_dup':
            return JoinOrderBenchmark.JOB_ORIGINAL_DUP_PRED.get(filepath,None)
        elif use_cols == 'uae-light':
            return JoinOrderBenchmark.UAE_PRED_COLS.get(filepath,None)

        assert False
        # return None  # Load all.

    if table:
        filepath = table + '.csv'
        table = TryLoad(
            table,
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
            # escapechar='\\',
        )
        return table

    tables = {}
    file_path_list=list()
    if use_cols == 'simple':
        file_path_list = JoinOrderBenchmark.BASE_TABLE_PRED_COLS
    elif use_cols == 'content':
        file_path_list = JoinOrderBenchmark.ContentColumns
    elif use_cols == 'multi':
        file_path_list = JoinOrderBenchmark.JOB_M_PRED_COLS
    elif use_cols == 'toy' :
        file_path_list = JoinOrderBenchmark.JOB_TOY_PRED_COLS
    elif use_cols =='imdb-db':
        file_path_list = JoinOrderBenchmark.IMDB_DB_PRED_COLS
    elif use_cols =='imdb-full-dup':
        file_path_list = JoinOrderBenchmark.IMDB_FULL_DUP_PRED_COLS
    elif use_cols =='union':
        file_path_list = JoinOrderBenchmark.JOB_UNION_PRED_COLS
    elif use_cols == 'deepdb':
        file_path_list = JoinOrderBenchmark.JOB_DEEPDB_PRED_COLS
    elif use_cols == 'imdb-full-deepdb':
        file_path_list = JoinOrderBenchmark.IMDB_DEEPDB_PRED_COLS
    elif use_cols == 'job':
        file_path_list = JoinOrderBenchmark.JOB_ORIGINAL_PRED_COLS
    elif use_cols =='original_dup':
        file_path_list = JoinOrderBenchmark.JOB_ORIGINAL_DUP_PRED
    elif use_cols == 'uae-light':
        file_path_list = JoinOrderBenchmark.UAE_PRED_COLS
    else :
        assert False
    for filepath in file_path_list:
        tables[filepath[0:-4]] = TryLoad(
            filepath[0:-4],
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
            # escapechar='\\',
        )

    return tables

class SyntheticSingleDataset(object):
    SYN_SINGLE_PRED_COLS = collections.defaultdict(
        list,{
            'table0.csv' : ['col0','col1','col2','col3','col4','col5','col6','col7','col8','col9'],
        })
    CSV_FILES = ['table0.csv']

    TRUE_FULL_OUTER_CARDINALITY = {
        # ('table0') : 1000000,
    }
    @staticmethod
    def GetFullOuterCardinalityOrFail(join_tables):
        key = tuple(sorted(join_tables))
        return SyntheticSingleDataset.TRUE_FULL_OUTER_CARDINALITY[key]

class SyntheticDataset(object):
    SYN_MULTI_PRED_COLS = collections.defaultdict(
        list,{
            'table0.csv' : ['PK'],
            'table1.csv' : ['PK','FK'],
            'table2.csv' : ['PK','FK'],
            'table3.csv' : ['PK','FK'],
            'table4.csv' : ['PK','FK'],
            'table5.csv' : ['PK','FK'],
            'table6.csv' : ['PK','FK'],
            'table7.csv' : ['PK','FK'],
            'table8.csv' : ['PK','FK'],
            'table9.csv' : ['PK','FK'],
        }
    )
    CSV_FILES = ['table0.csv', 'table1.csv', 'table2.csv', 'table3.csv', 'table4.csv', 'table5.csv', 'table6.csv', 'table7.csv', 'table8.csv', 'table9.csv']

    SYN_MULTI_REV_PRED_COLS = collections.defaultdict(
        list,{
            'table0.csv' : ['PK'],
            'table1.csv' : ['FK','PK'],
            'table2.csv' : ['FK','PK'],
            'table3.csv' : ['FK','PK'],
            'table4.csv' : ['FK','PK'],
            'table5.csv' : ['FK','PK'],
            'table6.csv' : ['FK','PK'],
            'table7.csv' : ['FK','PK'],
            'table8.csv' : ['FK','PK'],
            'table9.csv' : ['FK','PK'],
        }
    )
    TRUE_FULL_OUTER_CARDINALITY = {
        # ('table0', 'table1', 'table2', 'table3', 'table4', 'table5', 'table6', 'table7', 'table8', 'table9'):7814479,
    }
    @staticmethod
    def GetFullOuterCardinalityOrFail(join_tables):
        key = tuple(sorted(join_tables))
        return SyntheticDataset.TRUE_FULL_OUTER_CARDINALITY[key]
def LoadSYN(table=None,
              data_dir='',
              try_load_parsed=True,
              use_cols='multi'):
    assert use_cols in ['multi','single','multi_rev'], use_cols
    print(f"loadsyn from {data_dir}")

    def TryLoad(table_name, filepath, use_cols, **kwargs):
        """Try load from previously parsed (table, columns)."""
        if use_cols:
            cols_str = '-'.join(use_cols)
            parsed_path = filepath[:-4] + '.{}.table'.format(cols_str)
        else:
            parsed_path = filepath[:-4] + '.table'
        if try_load_parsed:
            if os.path.exists(parsed_path):
                arr = np.load(parsed_path, allow_pickle=True)
                print('Loaded parsed Table from', parsed_path)
                table = arr.item()
                print(table)
                return table
        table = CsvTable(
            table_name,
            filepath,
            cols=use_cols,
            **kwargs,
        )
        if try_load_parsed:
            np.save(open(parsed_path, 'wb'), table)
            print('Saved parsed Table to', parsed_path)
        return table

    def get_use_cols(filepath):
        if use_cols == 'multi':
            return SyntheticDataset.SYN_MULTI_PRED_COLS.get(filepath, None)
        if use_cols == 'multi_rev':
            return SyntheticDataset.SYN_MULTI_REV_PRED_COLS.get(filepath, None)
        if use_cols == 'single':
            return SyntheticSingleDataset.SYN_SINGLE_PRED_COLS.get(filepath, None)
        assert False
        # return None  # Load all.

    if table:
        filepath = table + '.csv'
        table = TryLoad(
            table,
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
        )
        return table

    tables = {}
    if use_cols == 'multi':
        file_path_list = SyntheticDataset.SYN_MULTI_PRED_COLS
    elif use_cols == 'multi_rev':
        file_path_list = SyntheticDataset.SYN_MULTI_REV_PRED_COLS
    elif use_cols == 'single':
        file_path_list = SyntheticSingleDataset.SYN_SINGLE_PRED_COLS
    else: assert False
    for filepath in file_path_list:
        tables[filepath[0:-4]] = TryLoad(
            filepath[0:-4],
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
        )
    return tables

class TPCDSBenchmark(object):
    ALIAS_TO_TABLE_NAME = {
        'ss': 'store_sales',
        'sr': 'store_returns',
        'cs': 'catalog_sales',
        'cr': 'catalog_returns',
        'ws': 'web_sales',
        'wr': 'web_returns',
        'inv': 'inventory',
        's': 'store',
        'cc': 'call_center',
        'cp': 'catalog_page',
        'web': 'web_site',
        'wp': 'web_page',
        'w': 'warehouse',
        'c': 'customer',
        'ca': 'customer_address',
        'cd': 'customer_demographics',
        'd': 'date_dim',
        'hd': 'household_demographics',
        'i': 'item',
        'ib': 'income_band',
        'p': 'promotion',
        'r': 'reason',
        'sm': 'ship_mode',
        't': 'time_dim',
    }
    CATEGORICAL_COLUMNS = collections.defaultdict(
        list,
        {
            'call_center': ['cc_call_center_id',  'cc_call_center_sk',  'cc_city',  'cc_class',  'cc_closed_date_sk',  'cc_company',  'cc_company_name',  'cc_country',  'cc_county',  'cc_division',  'cc_division_name',  'cc_employees',  'cc_gmt_offset',  'cc_hours',  'cc_manager',  'cc_market_manager',  'cc_mkt_class',  'cc_mkt_desc',  'cc_mkt_id',  'cc_name',  'cc_open_date_sk',  'cc_rec_end_date',  'cc_rec_start_date',  'cc_sq_ft',  'cc_state',  'cc_street_name',  'cc_street_number',  'cc_street_type',  'cc_suite_number',  'cc_tax_percentage',  'cc_zip'],
            'catalog_page': ['cp_catalog_number',  'cp_catalog_page_id',  'cp_catalog_page_number',  'cp_catalog_page_sk',  'cp_department',  'cp_description',  'cp_end_date_sk',  'cp_start_date_sk',  'cp_type'],
            'catalog_returns': ['cr_call_center_sk',  'cr_catalog_page_sk',  'cr_item_sk',  'cr_order_number',  'cr_reason_sk',  'cr_refunded_addr_sk',  'cr_refunded_cdemo_sk',  'cr_refunded_customer_sk',  'cr_refunded_hdemo_sk',  'cr_returned_date_sk',  'cr_returned_time_sk',  'cr_returning_addr_sk',  'cr_returning_cdemo_sk',  'cr_returning_customer_sk',  'cr_returning_hdemo_sk',  'cr_ship_mode_sk',  'cr_warehouse_sk'],
            'catalog_sales': ['cs_bill_addr_sk',  'cs_bill_cdemo_sk',  'cs_bill_customer_sk',  'cs_bill_hdemo_sk',  'cs_call_center_sk',  'cs_catalog_page_sk',  'cs_item_sk',  'cs_order_number',  'cs_promo_sk',  'cs_ship_addr_sk',  'cs_ship_cdemo_sk',  'cs_ship_customer_sk',  'cs_ship_hdemo_sk',  'cs_ship_mode_sk',  'cs_sold_time_sk',  'cs_warehouse_sk'],
            'customer': ['c_birth_country',  'c_birth_day',  'c_birth_month',  'c_birth_year',  'c_current_addr_sk',  'c_current_cdemo_sk',  'c_current_hdemo_sk',  'c_customer_id',  'c_customer_sk',  'c_email_address',  'c_first_name',  'c_first_sales_date_sk',  'c_first_shipto_date_sk',  'c_last_name',  'c_last_review_date_sk',  'c_login',  'c_preferred_cust_flag',  'c_salutation'],
            'customer_address': ['ca_address_id',  'ca_address_sk',  'ca_city',  'ca_country',  'ca_county',  'ca_gmt_offset',  'ca_location_type',  'ca_state',  'ca_street_name',  'ca_street_number',  'ca_street_type',  'ca_suite_number',  'ca_zip'],
            'customer_demographics': ['cd_credit_rating',  'cd_demo_sk',  'cd_education_status',  'cd_gender',  'cd_marital_status',  'cd_purchase_estimate'],
            'date_dim': ['d_current_day',  'd_current_month',  'd_current_quarter',  'd_current_week',  'd_current_year',  'd_day_name',  'd_dow',  'd_first_dom',  'd_following_holiday',  'd_fy_quarter_seq',  'd_fy_week_seq',  'd_fy_year',  'd_holiday',  'd_last_dom',  'd_quarter_name',  'd_quarter_seq',  'd_week_seq',  'd_weekend'],
            'household_demographics': ['hd_buy_potential',  'hd_demo_sk',  'hd_income_band_sk'],
            'income_band': ['ib_income_band_sk'],
            'inventory': ['inv_date_sk', 'inv_item_sk', 'inv_warehouse_sk'],
            'item': ['i_brand',  'i_brand_id',  'i_category',  'i_category_id',  'i_class',  'i_class_id',  'i_color',  'i_container',  'i_item_id',  'i_item_sk',  'i_manager_id',  'i_manufact',  'i_product_name',  'i_rec_end_date',  'i_rec_start_date',  'i_size',  'i_units',  'i_wholesale_cost'],
            'promotion': ['p_channel_catalog',  'p_channel_demo',  'p_channel_details',  'p_channel_dmail',  'p_channel_email',  'p_channel_event',  'p_channel_press',  'p_channel_radio',  'p_channel_tv',  'p_cost',  'p_discount_active',  'p_end_date_sk',  'p_item_sk',  'p_promo_id',  'p_promo_name',  'p_promo_sk',  'p_purpose',  'p_response_target',  'p_start_date_sk'],
            'reason': ['r_reason_desc', 'r_reason_id', 'r_reason_sk'],
            'ship_mode': ['sm_carrier',  'sm_code',  'sm_contract',  'sm_ship_mode_id',  'sm_ship_mode_sk',  'sm_type'],
            'store': ['s_city',  's_closed_date_sk',  's_company_id',  's_company_name',  's_country',  's_county',  's_division_id',  's_division_name',  's_floor_space',  's_geography_class',  's_gmt_offset',  's_hours',  's_manager',  's_market_desc',  's_market_id',  's_market_manager',  's_rec_end_date',  's_rec_start_date',  's_state',  's_store_id',  's_store_name',  's_store_sk',  's_street_name',  's_street_number',  's_street_type',  's_suite_number',  's_tax_precentage',  's_zip'],
            'store_returns': ['sr_addr_sk',  'sr_cdemo_sk',  'sr_customer_sk',  'sr_fee',  'sr_hdemo_sk',  'sr_item_sk',  'sr_reason_sk',  'sr_return_ship_cost',  'sr_return_time_sk',  'sr_store_sk',  'sr_ticket_number'],
            'store_sales': ['ss_addr_sk',  'ss_cdemo_sk',  'ss_customer_sk',  'ss_hdemo_sk',  'ss_item_sk',  'ss_promo_sk',  'ss_sold_time_sk',  'ss_store_sk',  'ss_ticket_number'],
            'time_dim': ['t_am_pm',  't_meal_time',  't_shift',  't_sub_shift',  't_time_id',  't_time_sk'],
            'warehouse': ['w_city',  'w_country',  'w_county',  'w_gmt_offset',  'w_state',  'w_street_name',  'w_street_number',  'w_street_type',  'w_suite_number',  'w_warehouse_id',  'w_warehouse_name',  'w_warehouse_sk',  'w_warehouse_sq_ft',  'w_zip'],
            'web_page': ['wp_access_date_sk',  'wp_autogen_flag',  'wp_creation_date_sk',  'wp_customer_sk',  'wp_image_count',  'wp_link_count',  'wp_max_ad_count',  'wp_rec_end_date',  'wp_rec_start_date',  'wp_type',  'wp_url',  'wp_web_page_id',  'wp_web_page_sk'],
            'web_returns': ['wr_item_sk',  'wr_order_number',  'wr_reason_sk',  'wr_refunded_addr_sk',  'wr_refunded_cdemo_sk',  'wr_refunded_customer_sk',  'wr_refunded_hdemo_sk',  'wr_returned_date_sk',  'wr_returned_time_sk',  'wr_returning_addr_sk',  'wr_returning_cdemo_sk',  'wr_returning_customer_sk',  'wr_returning_hdemo_sk',  'wr_web_page_sk'],
            'web_sales': ['ws_bill_addr_sk',  'ws_bill_cdemo_sk',  'ws_bill_customer_sk',  'ws_bill_hdemo_sk',  'ws_item_sk',  'ws_order_number',  'ws_promo_sk',  'ws_ship_addr_sk',  'ws_ship_cdemo_sk',  'ws_ship_customer_sk',  'ws_ship_hdemo_sk',  'ws_ship_mode_sk',  'ws_sold_time_sk',  'ws_warehouse_sk',  'ws_web_page_sk',  'ws_web_site_sk'],
            'web_site': ['web_city',  'web_class',  'web_close_date_sk',  'web_company_id',  'web_company_name',  'web_country',  'web_county',  'web_gmt_offset',  'web_manager',  'web_market_manager',  'web_mkt_class',  'web_mkt_desc',  'web_mkt_id',  'web_name',  'web_open_date_sk',  'web_rec_end_date',  'web_rec_start_date',  'web_site_id',  'web_site_sk',  'web_state',  'web_street_name',  'web_street_number',  'web_street_type',  'web_suite_number',  'web_tax_percentage',  'web_zip']
        })

    # Columns with a reasonable range/IN interpretation.
    RANGE_COLUMNS = collections.defaultdict(
        list,
        {
            'call_center': [],
            'catalog_page': [],
            'catalog_returns': ['cr_fee',  'cr_net_loss',  'cr_refunded_cash',  'cr_return_amount',  'cr_return_amt_inc_tax',  'cr_return_quantity',  'cr_return_ship_cost',  'cr_return_tax',  'cr_reversed_charge',  'cr_store_credit'],
            'catalog_sales': ['cs_coupon_amt',  'cs_ext_discount_amt',  'cs_ext_list_price',  'cs_ext_sales_price',  'cs_ext_ship_cost',  'cs_ext_tax',  'cs_ext_wholesale_cost',  'cs_list_price',  'cs_net_paid',  'cs_net_paid_inc_ship',  'cs_net_paid_inc_ship_tax',  'cs_net_paid_inc_tax',  'cs_net_profit',  'cs_quantity',  'cs_sales_price',  'cs_ship_date_sk',  'cs_sold_date_sk',  'cs_wholesale_cost'],
            'customer': [],
            'customer_address': [],
            'customer_demographics': ['cd_dep_college_count',  'cd_dep_count',  'cd_dep_employed_count'],
            'date_dim': ['d_date',  'd_date_id',  'd_date_sk',  'd_dom',  'd_month_seq',  'd_moy',  'd_qoy',  'd_same_day_lq',  'd_same_day_ly',  'd_year'],
            'household_demographics': ['hd_dep_count', 'hd_vehicle_count'],
            'income_band': ['ib_lower_bound', 'ib_upper_bound'],
            'inventory': ['inv_quantity_on_hand'],
            'item': ['i_current_price', 'i_formulation', 'i_item_desc', 'i_manufact_id'],
            'promotion': [],
            'reason': [],
            'ship_mode': [],
            'store': ['s_number_employees'],
            'store_returns': ['sr_net_loss',  'sr_refunded_cash',  'sr_return_amt',  'sr_return_amt_inc_tax',  'sr_return_quantity',  'sr_return_tax',  'sr_returned_date_sk',  'sr_reversed_charge',  'sr_store_credit'],
            'store_sales': ['ss_coupon_amt',  'ss_ext_discount_amt',  'ss_ext_list_price',  'ss_ext_sales_price',  'ss_ext_tax',  'ss_ext_wholesale_cost',  'ss_list_price',  'ss_net_paid',  'ss_net_paid_inc_tax',  'ss_net_profit',  'ss_quantity',  'ss_sales_price',  'ss_sold_date_sk',  'ss_wholesale_cost'],
            'time_dim': ['t_hour', 't_minute', 't_second', 't_time'],
            'warehouse': [],
            'web_page': ['wp_char_count'],
            'web_returns': ['wr_account_credit',  'wr_fee',  'wr_net_loss',  'wr_refunded_cash',  'wr_return_amt',  'wr_return_amt_inc_tax',  'wr_return_quantity',  'wr_return_ship_cost',  'wr_return_tax',  'wr_reversed_charge'],
            'web_sales': ['ws_coupon_amt',  'ws_ext_discount_amt',  'ws_ext_list_price',  'ws_ext_sales_price',  'ws_ext_ship_cost',  'ws_ext_tax',  'ws_ext_wholesale_cost',  'ws_list_price',  'ws_net_paid',  'ws_net_paid_inc_ship',  'ws_net_paid_inc_ship_tax',  'ws_net_paid_inc_tax',  'ws_net_profit',  'ws_quantity',  'ws_sales_price',  'ws_ship_date_sk',  'ws_sold_date_sk',  'ws_wholesale_cost'],
            'web_site': []
        })
    TPCDS_TOY_PRED_COLS = collections.defaultdict(list,{
            'store_returns.csv': ['sr_returned_date_sk', 'sr_return_time_sk', 'sr_item_sk', 'sr_customer_sk', 'sr_cdemo_sk', 'sr_hdemo_sk', 'sr_addr_sk', 'sr_store_sk', 'sr_reason_sk', 'sr_ticket_number', 'sr_return_quantity', 'sr_return_amt', 'sr_return_tax', 'sr_return_amt_inc_tax', 'sr_fee', 'sr_return_ship_cost', 'sr_refunded_cash', 'sr_reversed_charge', 'sr_store_credit', 'sr_net_loss'],
            'store_sales.csv': ['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk', 'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk', 'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price', 'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax', 'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'],
            'item.csv': ['i_item_sk', 'i_item_id', 'i_rec_start_date', 'i_rec_end_date', 'i_item_desc', 'i_current_price', 'i_wholesale_cost', 'i_brand_id', 'i_brand', 'i_class_id', 'i_class', 'i_category_id', 'i_category', 'i_manufact_id', 'i_manufact', 'i_size', 'i_formulation', 'i_color', 'i_units', 'i_container', 'i_manager_id', 'i_product_name'],
           })

    TPCDS_DB_PRED_COLS = collections.defaultdict(list,{
            'warehouse.csv': ['w_warehouse_sk', 'w_warehouse_id', 'w_warehouse_name', 'w_warehouse_sq_ft', 'w_street_number', 'w_street_name', 'w_street_type', 'w_suite_number', 'w_city', 'w_county', 'w_state', 'w_zip', 'w_country', 'w_gmt_offset'],
            'store_returns.csv': ['sr_returned_date_sk', 'sr_return_time_sk', 'sr_item_sk', 'sr_customer_sk', 'sr_cdemo_sk', 'sr_hdemo_sk', 'sr_addr_sk', 'sr_store_sk', 'sr_reason_sk', 'sr_ticket_number', 'sr_return_quantity', 'sr_return_amt', 'sr_return_tax', 'sr_return_amt_inc_tax', 'sr_fee', 'sr_return_ship_cost', 'sr_refunded_cash', 'sr_reversed_charge', 'sr_store_credit', 'sr_net_loss'],
            'household_demographics.csv': ['hd_demo_sk', 'hd_income_band_sk', 'hd_buy_potential', 'hd_dep_count', 'hd_vehicle_count'],
            'store_sales.csv': ['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk', 'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk', 'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price', 'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax', 'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'],
            'web_page.csv': ['wp_web_page_sk', 'wp_web_page_id', 'wp_rec_start_date', 'wp_rec_end_date', 'wp_creation_date_sk', 'wp_access_date_sk', 'wp_autogen_flag', 'wp_customer_sk', 'wp_url', 'wp_type', 'wp_char_count', 'wp_link_count', 'wp_image_count', 'wp_max_ad_count'],
            'customer.csv': ['c_customer_sk', 'c_customer_id', 'c_current_cdemo_sk', 'c_current_hdemo_sk', 'c_current_addr_sk', 'c_first_shipto_date_sk', 'c_first_sales_date_sk', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date_sk'],
            'item.csv': ['i_item_sk', 'i_item_id', 'i_rec_start_date', 'i_rec_end_date', 'i_item_desc', 'i_current_price', 'i_wholesale_cost', 'i_brand_id', 'i_brand', 'i_class_id', 'i_class', 'i_category_id', 'i_category', 'i_manufact_id', 'i_manufact', 'i_size', 'i_formulation', 'i_color', 'i_units', 'i_container', 'i_manager_id', 'i_product_name'],
            'web_site.csv': ['web_site_sk', 'web_site_id', 'web_rec_start_date', 'web_rec_end_date', 'web_name', 'web_open_date_sk', 'web_close_date_sk', 'web_class', 'web_manager', 'web_mkt_id', 'web_mkt_class', 'web_mkt_desc', 'web_market_manager', 'web_company_id', 'web_company_name', 'web_street_number', 'web_street_name', 'web_street_type', 'web_suite_number', 'web_city', 'web_county', 'web_state', 'web_zip', 'web_country', 'web_gmt_offset', 'web_tax_percentage'],
            'catalog_page.csv': ['cp_catalog_page_sk', 'cp_catalog_page_id', 'cp_start_date_sk', 'cp_end_date_sk', 'cp_department', 'cp_catalog_number', 'cp_catalog_page_number', 'cp_description', 'cp_type'],
            'customer_demographics.csv': ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'],
            'promotion.csv': ['p_promo_sk', 'p_promo_id', 'p_start_date_sk', 'p_end_date_sk', 'p_item_sk', 'p_cost', 'p_response_target', 'p_promo_name', 'p_channel_dmail', 'p_channel_email', 'p_channel_catalog', 'p_channel_tv', 'p_channel_radio', 'p_channel_press', 'p_channel_event', 'p_channel_demo', 'p_channel_details', 'p_purpose', 'p_discount_active'],
            'web_returns.csv': ['wr_returned_date_sk', 'wr_returned_time_sk', 'wr_item_sk', 'wr_refunded_customer_sk', 'wr_refunded_cdemo_sk', 'wr_refunded_hdemo_sk', 'wr_refunded_addr_sk', 'wr_returning_customer_sk', 'wr_returning_cdemo_sk', 'wr_returning_hdemo_sk', 'wr_returning_addr_sk', 'wr_web_page_sk', 'wr_reason_sk', 'wr_order_number', 'wr_return_quantity', 'wr_return_amt', 'wr_return_tax', 'wr_return_amt_inc_tax', 'wr_fee', 'wr_return_ship_cost', 'wr_refunded_cash', 'wr_reversed_charge', 'wr_account_credit', 'wr_net_loss'],
            'call_center.csv': ['cc_call_center_sk', 'cc_call_center_id', 'cc_rec_start_date', 'cc_rec_end_date', 'cc_closed_date_sk', 'cc_open_date_sk', 'cc_name', 'cc_class', 'cc_employees', 'cc_sq_ft', 'cc_hours', 'cc_manager', 'cc_mkt_id', 'cc_mkt_class', 'cc_mkt_desc', 'cc_market_manager', 'cc_division', 'cc_division_name', 'cc_company', 'cc_company_name', 'cc_street_number', 'cc_street_name', 'cc_street_type', 'cc_suite_number', 'cc_city', 'cc_county', 'cc_state', 'cc_zip', 'cc_country', 'cc_gmt_offset', 'cc_tax_percentage'],
            'date_dim.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'],
            'web_sales.csv': ['ws_sold_date_sk', 'ws_sold_time_sk', 'ws_ship_date_sk', 'ws_item_sk', 'ws_bill_customer_sk', 'ws_bill_cdemo_sk', 'ws_bill_hdemo_sk', 'ws_bill_addr_sk', 'ws_ship_customer_sk', 'ws_ship_cdemo_sk', 'ws_ship_hdemo_sk', 'ws_ship_addr_sk', 'ws_web_page_sk', 'ws_web_site_sk', 'ws_ship_mode_sk', 'ws_warehouse_sk', 'ws_promo_sk', 'ws_order_number', 'ws_quantity', 'ws_wholesale_cost', 'ws_list_price', 'ws_sales_price', 'ws_ext_discount_amt', 'ws_ext_sales_price', 'ws_ext_wholesale_cost', 'ws_ext_list_price', 'ws_ext_tax', 'ws_coupon_amt', 'ws_ext_ship_cost', 'ws_net_paid', 'ws_net_paid_inc_tax', 'ws_net_paid_inc_ship', 'ws_net_paid_inc_ship_tax', 'ws_net_profit'],
            'customer_address.csv': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'],
            'catalog_sales.csv': ['cs_sold_date_sk', 'cs_sold_time_sk', 'cs_ship_date_sk', 'cs_bill_customer_sk', 'cs_bill_cdemo_sk', 'cs_bill_hdemo_sk', 'cs_bill_addr_sk', 'cs_ship_customer_sk', 'cs_ship_cdemo_sk', 'cs_ship_hdemo_sk', 'cs_ship_addr_sk', 'cs_call_center_sk', 'cs_catalog_page_sk', 'cs_ship_mode_sk', 'cs_warehouse_sk', 'cs_item_sk', 'cs_promo_sk', 'cs_order_number', 'cs_quantity', 'cs_wholesale_cost', 'cs_list_price', 'cs_sales_price', 'cs_ext_discount_amt', 'cs_ext_sales_price', 'cs_ext_wholesale_cost', 'cs_ext_list_price', 'cs_ext_tax', 'cs_coupon_amt', 'cs_ext_ship_cost', 'cs_net_paid', 'cs_net_paid_inc_tax', 'cs_net_paid_inc_ship', 'cs_net_paid_inc_ship_tax', 'cs_net_profit'],
            'reason.csv': ['r_reason_sk', 'r_reason_id', 'r_reason_desc'],
            'catalog_returns.csv': ['cr_returned_date_sk', 'cr_returned_time_sk', 'cr_item_sk', 'cr_refunded_customer_sk', 'cr_refunded_cdemo_sk', 'cr_refunded_hdemo_sk', 'cr_refunded_addr_sk', 'cr_returning_customer_sk', 'cr_returning_cdemo_sk', 'cr_returning_hdemo_sk', 'cr_returning_addr_sk', 'cr_call_center_sk', 'cr_catalog_page_sk', 'cr_ship_mode_sk', 'cr_warehouse_sk', 'cr_reason_sk', 'cr_order_number', 'cr_return_quantity', 'cr_return_amount', 'cr_return_tax', 'cr_return_amt_inc_tax', 'cr_fee', 'cr_return_ship_cost', 'cr_refunded_cash', 'cr_reversed_charge', 'cr_store_credit', 'cr_net_loss'],
            'time_dim.csv': ['t_time_sk', 't_time_id', 't_time', 't_hour', 't_minute', 't_second', 't_am_pm', 't_shift', 't_sub_shift', 't_meal_time'],
            'ship_mode.csv': ['sm_ship_mode_sk', 'sm_ship_mode_id', 'sm_type', 'sm_code', 'sm_carrier', 'sm_contract'],
            'income_band.csv': ['ib_income_band_sk', 'ib_lower_bound', 'ib_upper_bound'],
            'store.csv': ['s_store_sk', 's_store_id', 's_rec_start_date', 's_rec_end_date', 's_closed_date_sk', 's_store_name', 's_number_employees', 's_floor_space', 's_hours', 's_manager', 's_market_id', 's_geography_class', 's_market_desc', 's_market_manager', 's_division_id', 's_division_name', 's_company_id', 's_company_name', 's_street_number', 's_street_name', 's_street_type', 's_suite_number', 's_city', 's_county', 's_state', 's_zip', 's_country', 's_gmt_offset', 's_tax_precentage'],
            'inventory.csv': ['inv_date_sk', 'inv_item_sk', 'inv_warehouse_sk', 'inv_quantity_on_hand'],
    })
    TPCDS_DB_DUP_PRED_COLS = collections.defaultdict(list,{'warehouse.csv': ['w_warehouse_sk', 'w_warehouse_id', 'w_warehouse_name', 'w_warehouse_sq_ft', 'w_street_number', 'w_street_name', 'w_street_type', 'w_suite_number', 'w_city', 'w_county', 'w_state', 'w_zip', 'w_country', 'w_gmt_offset'], 'store_returns.csv': ['sr_returned_date_sk', 'sr_return_time_sk', 'sr_item_sk', 'sr_customer_sk', 'sr_cdemo_sk', 'sr_hdemo_sk', 'sr_addr_sk', 'sr_store_sk', 'sr_reason_sk', 'sr_ticket_number', 'sr_return_quantity', 'sr_return_amt', 'sr_return_tax', 'sr_return_amt_inc_tax', 'sr_fee', 'sr_return_ship_cost', 'sr_refunded_cash', 'sr_reversed_charge', 'sr_store_credit', 'sr_net_loss'], 'household_demographics.csv': ['hd_demo_sk', 'hd_income_band_sk', 'hd_buy_potential', 'hd_dep_count', 'hd_vehicle_count'], 'store_sales.csv': ['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk', 'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk', 'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price', 'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax', 'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'], 'web_page.csv': ['wp_web_page_sk', 'wp_web_page_id', 'wp_rec_start_date', 'wp_rec_end_date', 'wp_creation_date_sk', 'wp_access_date_sk', 'wp_autogen_flag', 'wp_customer_sk', 'wp_url', 'wp_type', 'wp_char_count', 'wp_link_count', 'wp_image_count', 'wp_max_ad_count'], 'customer.csv': ['c_customer_sk', 'c_customer_id', 'c_current_cdemo_sk', 'c_current_hdemo_sk', 'c_current_addr_sk', 'c_first_shipto_date_sk', 'c_first_sales_date_sk', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date_sk'], 'item.csv': ['i_item_sk', 'i_item_id', 'i_rec_start_date', 'i_rec_end_date', 'i_item_desc', 'i_current_price', 'i_wholesale_cost', 'i_brand_id', 'i_brand', 'i_class_id', 'i_class', 'i_category_id', 'i_category', 'i_manufact_id', 'i_manufact', 'i_size', 'i_formulation', 'i_color', 'i_units', 'i_container', 'i_manager_id', 'i_product_name'], 'web_site.csv': ['web_site_sk', 'web_site_id', 'web_rec_start_date', 'web_rec_end_date', 'web_name', 'web_open_date_sk', 'web_close_date_sk', 'web_class', 'web_manager', 'web_mkt_id', 'web_mkt_class', 'web_mkt_desc', 'web_market_manager', 'web_company_id', 'web_company_name', 'web_street_number', 'web_street_name', 'web_street_type', 'web_suite_number', 'web_city', 'web_county', 'web_state', 'web_zip', 'web_country', 'web_gmt_offset', 'web_tax_percentage'], 'catalog_page.csv': ['cp_catalog_page_sk', 'cp_catalog_page_id', 'cp_start_date_sk', 'cp_end_date_sk', 'cp_department', 'cp_catalog_number', 'cp_catalog_page_number', 'cp_description', 'cp_type'], 'customer_demographics.csv': ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'], 'promotion.csv': ['p_promo_sk', 'p_promo_id', 'p_start_date_sk', 'p_end_date_sk', 'p_item_sk', 'p_cost', 'p_response_target', 'p_promo_name', 'p_channel_dmail', 'p_channel_email', 'p_channel_catalog', 'p_channel_tv', 'p_channel_radio', 'p_channel_press', 'p_channel_event', 'p_channel_demo', 'p_channel_details', 'p_purpose', 'p_discount_active'], 'web_returns.csv': ['wr_returned_date_sk', 'wr_returned_time_sk', 'wr_item_sk', 'wr_refunded_customer_sk', 'wr_refunded_cdemo_sk', 'wr_refunded_hdemo_sk', 'wr_refunded_addr_sk', 'wr_returning_customer_sk', 'wr_returning_cdemo_sk', 'wr_returning_hdemo_sk', 'wr_returning_addr_sk', 'wr_web_page_sk', 'wr_reason_sk', 'wr_order_number', 'wr_return_quantity', 'wr_return_amt', 'wr_return_tax', 'wr_return_amt_inc_tax', 'wr_fee', 'wr_return_ship_cost', 'wr_refunded_cash', 'wr_reversed_charge', 'wr_account_credit', 'wr_net_loss'], 'call_center.csv': ['cc_call_center_sk', 'cc_call_center_id', 'cc_rec_start_date', 'cc_rec_end_date', 'cc_closed_date_sk', 'cc_open_date_sk', 'cc_name', 'cc_class', 'cc_employees', 'cc_sq_ft', 'cc_hours', 'cc_manager', 'cc_mkt_id', 'cc_mkt_class', 'cc_mkt_desc', 'cc_market_manager', 'cc_division', 'cc_division_name', 'cc_company', 'cc_company_name', 'cc_street_number', 'cc_street_name', 'cc_street_type', 'cc_suite_number', 'cc_city', 'cc_county', 'cc_state', 'cc_zip', 'cc_country', 'cc_gmt_offset', 'cc_tax_percentage'], 'date_dim.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'web_sales.csv': ['ws_sold_date_sk', 'ws_sold_time_sk', 'ws_ship_date_sk', 'ws_item_sk', 'ws_bill_customer_sk', 'ws_bill_cdemo_sk', 'ws_bill_hdemo_sk', 'ws_bill_addr_sk', 'ws_ship_customer_sk', 'ws_ship_cdemo_sk', 'ws_ship_hdemo_sk', 'ws_ship_addr_sk', 'ws_web_page_sk', 'ws_web_site_sk', 'ws_ship_mode_sk', 'ws_warehouse_sk', 'ws_promo_sk', 'ws_order_number', 'ws_quantity', 'ws_wholesale_cost', 'ws_list_price', 'ws_sales_price', 'ws_ext_discount_amt', 'ws_ext_sales_price', 'ws_ext_wholesale_cost', 'ws_ext_list_price', 'ws_ext_tax', 'ws_coupon_amt', 'ws_ext_ship_cost', 'ws_net_paid', 'ws_net_paid_inc_tax', 'ws_net_paid_inc_ship', 'ws_net_paid_inc_ship_tax', 'ws_net_profit'], 'customer_address.csv': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'], 'catalog_sales.csv': ['cs_sold_date_sk', 'cs_sold_time_sk', 'cs_ship_date_sk', 'cs_bill_customer_sk', 'cs_bill_cdemo_sk', 'cs_bill_hdemo_sk', 'cs_bill_addr_sk', 'cs_ship_customer_sk', 'cs_ship_cdemo_sk', 'cs_ship_hdemo_sk', 'cs_ship_addr_sk', 'cs_call_center_sk', 'cs_catalog_page_sk', 'cs_ship_mode_sk', 'cs_warehouse_sk', 'cs_item_sk', 'cs_promo_sk', 'cs_order_number', 'cs_quantity', 'cs_wholesale_cost', 'cs_list_price', 'cs_sales_price', 'cs_ext_discount_amt', 'cs_ext_sales_price', 'cs_ext_wholesale_cost', 'cs_ext_list_price', 'cs_ext_tax', 'cs_coupon_amt', 'cs_ext_ship_cost', 'cs_net_paid', 'cs_net_paid_inc_tax', 'cs_net_paid_inc_ship', 'cs_net_paid_inc_ship_tax', 'cs_net_profit'], 'reason.csv': ['r_reason_sk', 'r_reason_id', 'r_reason_desc'], 'catalog_returns.csv': ['cr_returned_date_sk', 'cr_returned_time_sk', 'cr_item_sk', 'cr_refunded_customer_sk', 'cr_refunded_cdemo_sk', 'cr_refunded_hdemo_sk', 'cr_refunded_addr_sk', 'cr_returning_customer_sk', 'cr_returning_cdemo_sk', 'cr_returning_hdemo_sk', 'cr_returning_addr_sk', 'cr_call_center_sk', 'cr_catalog_page_sk', 'cr_ship_mode_sk', 'cr_warehouse_sk', 'cr_reason_sk', 'cr_order_number', 'cr_return_quantity', 'cr_return_amount', 'cr_return_tax', 'cr_return_amt_inc_tax', 'cr_fee', 'cr_return_ship_cost', 'cr_refunded_cash', 'cr_reversed_charge', 'cr_store_credit', 'cr_net_loss'], 'time_dim.csv': ['t_time_sk', 't_time_id', 't_time', 't_hour', 't_minute', 't_second', 't_am_pm', 't_shift', 't_sub_shift', 't_meal_time'], 'ship_mode.csv': ['sm_ship_mode_sk', 'sm_ship_mode_id', 'sm_type', 'sm_code', 'sm_carrier', 'sm_contract'], 'income_band.csv': ['ib_income_band_sk', 'ib_lower_bound', 'ib_upper_bound'], 'store.csv': ['s_store_sk', 's_store_id', 's_rec_start_date', 's_rec_end_date', 's_closed_date_sk', 's_store_name', 's_number_employees', 's_floor_space', 's_hours', 's_manager', 's_market_id', 's_geography_class', 's_market_desc', 's_market_manager', 's_division_id', 's_division_name', 's_company_id', 's_company_name', 's_street_number', 's_street_name', 's_street_type', 's_suite_number', 's_city', 's_county', 's_state', 's_zip', 's_country', 's_gmt_offset', 's_tax_precentage'], 'inventory.csv': ['inv_date_sk', 'inv_item_sk', 'inv_warehouse_sk', 'inv_quantity_on_hand'], 'catalog_sales_dup_1_.csv': ['cs_sold_date_sk', 'cs_sold_time_sk', 'cs_ship_date_sk', 'cs_bill_customer_sk', 'cs_bill_cdemo_sk', 'cs_bill_hdemo_sk', 'cs_bill_addr_sk', 'cs_ship_customer_sk', 'cs_ship_cdemo_sk', 'cs_ship_hdemo_sk', 'cs_ship_addr_sk', 'cs_call_center_sk', 'cs_catalog_page_sk', 'cs_ship_mode_sk', 'cs_warehouse_sk', 'cs_item_sk', 'cs_promo_sk', 'cs_order_number', 'cs_quantity', 'cs_wholesale_cost', 'cs_list_price', 'cs_sales_price', 'cs_ext_discount_amt', 'cs_ext_sales_price', 'cs_ext_wholesale_cost', 'cs_ext_list_price', 'cs_ext_tax', 'cs_coupon_amt', 'cs_ext_ship_cost', 'cs_net_paid', 'cs_net_paid_inc_tax', 'cs_net_paid_inc_ship', 'cs_net_paid_inc_ship_tax', 'cs_net_profit'], 'catalog_sales_dup_2_.csv': ['cs_sold_date_sk', 'cs_sold_time_sk', 'cs_ship_date_sk', 'cs_bill_customer_sk', 'cs_bill_cdemo_sk', 'cs_bill_hdemo_sk', 'cs_bill_addr_sk', 'cs_ship_customer_sk', 'cs_ship_cdemo_sk', 'cs_ship_hdemo_sk', 'cs_ship_addr_sk', 'cs_call_center_sk', 'cs_catalog_page_sk', 'cs_ship_mode_sk', 'cs_warehouse_sk', 'cs_item_sk', 'cs_promo_sk', 'cs_order_number', 'cs_quantity', 'cs_wholesale_cost', 'cs_list_price', 'cs_sales_price', 'cs_ext_discount_amt', 'cs_ext_sales_price', 'cs_ext_wholesale_cost', 'cs_ext_list_price', 'cs_ext_tax', 'cs_coupon_amt', 'cs_ext_ship_cost', 'cs_net_paid', 'cs_net_paid_inc_tax', 'cs_net_paid_inc_ship', 'cs_net_paid_inc_ship_tax', 'cs_net_profit'], 'customer_address_dup_1_.csv': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'], 'customer_address_dup_2_.csv': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'], 'customer_dup_1_.csv': ['c_customer_sk', 'c_customer_id', 'c_current_cdemo_sk', 'c_current_hdemo_sk', 'c_current_addr_sk', 'c_first_shipto_date_sk', 'c_first_sales_date_sk', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date_sk'], 'call_center_dup_1_.csv': ['cc_call_center_sk', 'cc_call_center_id', 'cc_rec_start_date', 'cc_rec_end_date', 'cc_closed_date_sk', 'cc_open_date_sk', 'cc_name', 'cc_class', 'cc_employees', 'cc_sq_ft', 'cc_hours', 'cc_manager', 'cc_mkt_id', 'cc_mkt_class', 'cc_mkt_desc', 'cc_market_manager', 'cc_division', 'cc_division_name', 'cc_company', 'cc_company_name', 'cc_street_number', 'cc_street_name', 'cc_street_type', 'cc_suite_number', 'cc_city', 'cc_county', 'cc_state', 'cc_zip', 'cc_country', 'cc_gmt_offset', 'cc_tax_percentage'], 'catalog_page_dup_1_.csv': ['cp_catalog_page_sk', 'cp_catalog_page_id', 'cp_start_date_sk', 'cp_end_date_sk', 'cp_department', 'cp_catalog_number', 'cp_catalog_page_number', 'cp_description', 'cp_type'], 'customer_address_dup_3_.csv': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'], 'customer_demographics_dup_1_.csv': ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'], 'customer_dup_2_.csv': ['c_customer_sk', 'c_customer_id', 'c_current_cdemo_sk', 'c_current_hdemo_sk', 'c_current_addr_sk', 'c_first_shipto_date_sk', 'c_first_sales_date_sk', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date_sk'], 'date_dim_dup_1_.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'date_dim_dup_2_.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'customer_address_dup_4_.csv': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'], 'customer_demographics_dup_2_.csv': ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'], 'household_demographics_dup_1_.csv': ['hd_demo_sk', 'hd_income_band_sk', 'hd_buy_potential', 'hd_dep_count', 'hd_vehicle_count'], 'date_dim_dup_3_.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'date_dim_dup_4_.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'date_dim_dup_5_.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'date_dim_dup_6_.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'warehouse_dup_1_.csv': ['w_warehouse_sk', 'w_warehouse_id', 'w_warehouse_name', 'w_warehouse_sq_ft', 'w_street_number', 'w_street_name', 'w_street_type', 'w_suite_number', 'w_city', 'w_county', 'w_state', 'w_zip', 'w_country', 'w_gmt_offset'], 'customer_demographics_dup_3_.csv': ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'], 'customer_dup_3_.csv': ['c_customer_sk', 'c_customer_id', 'c_current_cdemo_sk', 'c_current_hdemo_sk', 'c_current_addr_sk', 'c_first_shipto_date_sk', 'c_first_sales_date_sk', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date_sk'], 'store_sales_dup_1_.csv': ['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk', 'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk', 'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price', 'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax', 'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'], 'date_dim_dup_7_.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'store_sales_dup_2_.csv': ['ss_sold_date_sk', 'ss_sold_time_sk', 'ss_item_sk', 'ss_customer_sk', 'ss_cdemo_sk', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_store_sk', 'ss_promo_sk', 'ss_ticket_number', 'ss_quantity', 'ss_wholesale_cost', 'ss_list_price', 'ss_sales_price', 'ss_ext_discount_amt', 'ss_ext_sales_price', 'ss_ext_wholesale_cost', 'ss_ext_list_price', 'ss_ext_tax', 'ss_coupon_amt', 'ss_net_paid', 'ss_net_paid_inc_tax', 'ss_net_profit'], 'customer_address_dup_5_.csv': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'], 'customer_demographics_dup_4_.csv': ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'], 'customer_dup_4_.csv': ['c_customer_sk', 'c_customer_id', 'c_current_cdemo_sk', 'c_current_hdemo_sk', 'c_current_addr_sk', 'c_first_shipto_date_sk', 'c_first_sales_date_sk', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date_sk'], 'household_demographics_dup_2_.csv': ['hd_demo_sk', 'hd_income_band_sk', 'hd_buy_potential', 'hd_dep_count', 'hd_vehicle_count'], 'promotion_dup_1_.csv': ['p_promo_sk', 'p_promo_id', 'p_start_date_sk', 'p_end_date_sk', 'p_item_sk', 'p_cost', 'p_response_target', 'p_promo_name', 'p_channel_dmail', 'p_channel_email', 'p_channel_catalog', 'p_channel_tv', 'p_channel_radio', 'p_channel_press', 'p_channel_event', 'p_channel_demo', 'p_channel_details', 'p_purpose', 'p_discount_active'], 'date_dim_dup_8_.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'time_dim_dup_1_.csv': ['t_time_sk', 't_time_id', 't_time', 't_hour', 't_minute', 't_second', 't_am_pm', 't_shift', 't_sub_shift', 't_meal_time'], 'store_dup_1_.csv': ['s_store_sk', 's_store_id', 's_rec_start_date', 's_rec_end_date', 's_closed_date_sk', 's_store_name', 's_number_employees', 's_floor_space', 's_hours', 's_manager', 's_market_id', 's_geography_class', 's_market_desc', 's_market_manager', 's_division_id', 's_division_name', 's_company_id', 's_company_name', 's_street_number', 's_street_name', 's_street_type', 's_suite_number', 's_city', 's_county', 's_state', 's_zip', 's_country', 's_gmt_offset', 's_tax_precentage'], 'web_sales_dup_1_.csv': ['ws_sold_date_sk', 'ws_sold_time_sk', 'ws_ship_date_sk', 'ws_item_sk', 'ws_bill_customer_sk', 'ws_bill_cdemo_sk', 'ws_bill_hdemo_sk', 'ws_bill_addr_sk', 'ws_ship_customer_sk', 'ws_ship_cdemo_sk', 'ws_ship_hdemo_sk', 'ws_ship_addr_sk', 'ws_web_page_sk', 'ws_web_site_sk', 'ws_ship_mode_sk', 'ws_warehouse_sk', 'ws_promo_sk', 'ws_order_number', 'ws_quantity', 'ws_wholesale_cost', 'ws_list_price', 'ws_sales_price', 'ws_ext_discount_amt', 'ws_ext_sales_price', 'ws_ext_wholesale_cost', 'ws_ext_list_price', 'ws_ext_tax', 'ws_coupon_amt', 'ws_ext_ship_cost', 'ws_net_paid', 'ws_net_paid_inc_tax', 'ws_net_paid_inc_ship', 'ws_net_paid_inc_ship_tax', 'ws_net_profit'], 'web_sales_dup_2_.csv': ['ws_sold_date_sk', 'ws_sold_time_sk', 'ws_ship_date_sk', 'ws_item_sk', 'ws_bill_customer_sk', 'ws_bill_cdemo_sk', 'ws_bill_hdemo_sk', 'ws_bill_addr_sk', 'ws_ship_customer_sk', 'ws_ship_cdemo_sk', 'ws_ship_hdemo_sk', 'ws_ship_addr_sk', 'ws_web_page_sk', 'ws_web_site_sk', 'ws_ship_mode_sk', 'ws_warehouse_sk', 'ws_promo_sk', 'ws_order_number', 'ws_quantity', 'ws_wholesale_cost', 'ws_list_price', 'ws_sales_price', 'ws_ext_discount_amt', 'ws_ext_sales_price', 'ws_ext_wholesale_cost', 'ws_ext_list_price', 'ws_ext_tax', 'ws_coupon_amt', 'ws_ext_ship_cost', 'ws_net_paid', 'ws_net_paid_inc_tax', 'ws_net_paid_inc_ship', 'ws_net_paid_inc_ship_tax', 'ws_net_profit'], 'reason_dup_1_.csv': ['r_reason_sk', 'r_reason_id', 'r_reason_desc'], 'customer_address_dup_6_.csv': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'], 'customer_demographics_dup_5_.csv': ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'], 'date_dim_dup_9_.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'customer_address_dup_7_.csv': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'], 'customer_demographics_dup_6_.csv': ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'], 'customer_dup_5_.csv': ['c_customer_sk', 'c_customer_id', 'c_current_cdemo_sk', 'c_current_hdemo_sk', 'c_current_addr_sk', 'c_first_shipto_date_sk', 'c_first_sales_date_sk', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date_sk'], 'customer_address_dup_8_.csv': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'], 'customer_demographics_dup_7_.csv': ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'], 'customer_dup_6_.csv': ['c_customer_sk', 'c_customer_id', 'c_current_cdemo_sk', 'c_current_hdemo_sk', 'c_current_addr_sk', 'c_first_shipto_date_sk', 'c_first_sales_date_sk', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date_sk'], 'promotion_dup_2_.csv': ['p_promo_sk', 'p_promo_id', 'p_start_date_sk', 'p_end_date_sk', 'p_item_sk', 'p_cost', 'p_response_target', 'p_promo_name', 'p_channel_dmail', 'p_channel_email', 'p_channel_catalog', 'p_channel_tv', 'p_channel_radio', 'p_channel_press', 'p_channel_event', 'p_channel_demo', 'p_channel_details', 'p_purpose', 'p_discount_active'], 'customer_address_dup_9_.csv': ['ca_address_sk', 'ca_address_id', 'ca_street_number', 'ca_street_name', 'ca_street_type', 'ca_suite_number', 'ca_city', 'ca_county', 'ca_state', 'ca_zip', 'ca_country', 'ca_gmt_offset', 'ca_location_type'], 'customer_demographics_dup_8_.csv': ['cd_demo_sk', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_count', 'cd_dep_employed_count', 'cd_dep_college_count'], 'customer_dup_7_.csv': ['c_customer_sk', 'c_customer_id', 'c_current_cdemo_sk', 'c_current_hdemo_sk', 'c_current_addr_sk', 'c_first_shipto_date_sk', 'c_first_sales_date_sk', 'c_salutation', 'c_first_name', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_day', 'c_birth_month', 'c_birth_year', 'c_birth_country', 'c_login', 'c_email_address', 'c_last_review_date_sk'], 'date_dim_dup_10_.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'household_demographics_dup_3_.csv': ['hd_demo_sk', 'hd_income_band_sk', 'hd_buy_potential', 'hd_dep_count', 'hd_vehicle_count'], 'ship_mode_dup_1_.csv': ['sm_ship_mode_sk', 'sm_ship_mode_id', 'sm_type', 'sm_code', 'sm_carrier', 'sm_contract'], 'date_dim_dup_11_.csv': ['d_date_sk', 'd_date_id', 'd_date', 'd_month_seq', 'd_week_seq', 'd_quarter_seq', 'd_year', 'd_dow', 'd_moy', 'd_dom', 'd_qoy', 'd_fy_year', 'd_fy_quarter_seq', 'd_fy_week_seq', 'd_day_name', 'd_quarter_name', 'd_holiday', 'd_weekend', 'd_following_holiday', 'd_first_dom', 'd_last_dom', 'd_same_day_ly', 'd_same_day_lq', 'd_current_day', 'd_current_week', 'd_current_month', 'd_current_quarter', 'd_current_year'], 'time_dim_dup_2_.csv': ['t_time_sk', 't_time_id', 't_time', 't_hour', 't_minute', 't_second', 't_am_pm', 't_shift', 't_sub_shift', 't_meal_time'], 'warehouse_dup_2_.csv': ['w_warehouse_sk', 'w_warehouse_id', 'w_warehouse_name', 'w_warehouse_sq_ft', 'w_street_number', 'w_street_name', 'w_street_type', 'w_suite_number', 'w_city', 'w_county', 'w_state', 'w_zip', 'w_country', 'w_gmt_offset'], 'web_page_dup_1_.csv': ['wp_web_page_sk', 'wp_web_page_id', 'wp_rec_start_date', 'wp_rec_end_date', 'wp_creation_date_sk', 'wp_access_date_sk', 'wp_autogen_flag', 'wp_customer_sk', 'wp_url', 'wp_type', 'wp_char_count', 'wp_link_count', 'wp_image_count', 'wp_max_ad_count']})
    TPCDS_BENCHMARK_DUP_PRED_COLS = collections.defaultdict(list,{'catalog_returns.csv': ['cr_return_amount', 'cr_order_number', 'cr_returning_addr_sk', 'cr_call_center_sk', 'cr_net_loss', 'cr_reversed_charge', 'cr_store_credit', 'cr_catalog_page_sk', 'cr_return_quantity', 'cr_returned_date_sk', 'cr_returning_customer_sk', 'cr_return_amt_inc_tax', 'cr_refunded_addr_sk', 'cr_item_sk', 'cr_refunded_cash'], 'customer.csv': ['c_first_sales_date_sk', 'c_current_cdemo_sk', 'c_last_review_date_sk', 'c_current_hdemo_sk', 'c_customer_id', 'c_birth_month', 'c_birth_year', 'c_first_shipto_date_sk', 'c_birth_day', 'c_salutation', 'c_login', 'c_email_address', 'c_customer_sk', 'c_current_addr_sk', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_country', 'c_first_name'], 'customer_address.csv': ['ca_street_name', 'ca_gmt_offset', 'ca_zip', 'ca_location_type', 'ca_suite_number', 'ca_country', 'ca_street_number', 'ca_city', 'ca_county', 'ca_address_sk', 'ca_street_type', 'ca_state'], 'catalog_sales.csv': ['cs_sales_price', 'cs_ext_sales_price', 'cs_warehouse_sk', 'cs_ext_discount_amt', 'cs_net_profit', 'cs_bill_cdemo_sk', 'cs_promo_sk', 'cs_item_sk', 'cs_bill_hdemo_sk', 'cs_ship_customer_sk', 'cs_quantity', 'cs_wholesale_cost', 'cs_ext_ship_cost', 'cs_coupon_amt', 'cs_ext_list_price', 'cs_ship_cdemo_sk', 'cs_sold_time_sk', 'cs_net_paid', 'cs_ship_addr_sk', 'cs_ext_wholesale_cost', 'cs_list_price', 'cs_order_number', 'cs_catalog_page_sk', 'cs_bill_customer_sk', 'cs_sold_date_sk', 'cs_ship_date_sk', 'cs_ship_mode_sk', 'cs_call_center_sk', 'cs_bill_addr_sk'], 'warehouse.csv': ['w_warehouse_name', 'w_warehouse_sq_ft', 'w_country', 'w_state', 'w_county', 'w_warehouse_sk', 'w_city'], 'store_returns.csv': ['sr_reason_sk', 'sr_returned_date_sk', 'sr_return_amt', 'sr_item_sk', 'sr_customer_sk', 'sr_return_quantity', 'sr_cdemo_sk', 'sr_store_sk', 'sr_ticket_number', 'sr_net_loss'], 'customer_demographics.csv': ['cd_dep_count', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_employed_count', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_demo_sk', 'cd_dep_college_count'], 'web_returns.csv': ['wr_fee', 'wr_returned_date_sk', 'wr_reason_sk', 'wr_web_page_sk', 'wr_item_sk', 'wr_returning_cdemo_sk', 'wr_return_quantity', 'wr_return_amt', 'wr_refunded_addr_sk', 'wr_order_number', 'wr_returning_addr_sk', 'wr_net_loss', 'wr_refunded_cash', 'wr_refunded_cdemo_sk', 'wr_returning_customer_sk'], 'inventory.csv': ['inv_quantity_on_hand', 'inv_item_sk', 'inv_warehouse_sk', 'inv_date_sk'], 'web_sales.csv': ['ws_web_page_sk', 'ws_bill_addr_sk', 'ws_item_sk', 'ws_ext_discount_amt', 'ws_ext_sales_price', 'ws_net_paid', 'ws_ext_wholesale_cost', 'ws_wholesale_cost', 'ws_ship_hdemo_sk', 'ws_ship_customer_sk', 'ws_ship_addr_sk', 'ws_bill_customer_sk', 'ws_ship_date_sk', 'ws_net_profit', 'ws_sold_time_sk', 'ws_bill_cdemo_sk', 'ws_warehouse_sk', 'ws_sales_price', 'ws_sold_date_sk', 'ws_order_number', 'ws_promo_sk', 'ws_list_price', 'ws_web_site_sk', 'ws_ext_ship_cost', 'ws_ship_cdemo_sk', 'ws_ship_mode_sk', 'ws_quantity', 'ws_ext_list_price'], 'date_dim.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'store_sales.csv': ['ss_cdemo_sk', 'ss_coupon_amt', 'ss_ext_list_price', 'ss_ext_sales_price', 'ss_item_sk', 'ss_store_sk', 'ss_ext_discount_amt', 'ss_sold_time_sk', 'ss_sold_date_sk', 'ss_ext_tax', 'ss_customer_sk', 'ss_net_profit', 'ss_ext_wholesale_cost', 'ss_wholesale_cost', 'ss_ticket_number', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_promo_sk', 'ss_list_price', 'ss_sales_price', 'ss_net_paid', 'ss_quantity'], 'item.csv': ['i_manager_id', 'i_wholesale_cost', 'i_brand', 'i_manufact', 'i_size', 'i_category', 'i_item_sk', 'i_class', 'i_current_price', 'i_brand_id', 'i_item_desc', 'i_category_id', 'i_item_id', 'i_product_name', 'i_color', 'i_manufact_id', 'i_units', 'i_class_id'], 'store.csv': ['s_market_id', 's_zip', 's_company_name', 's_number_employees', 's_company_id', 's_store_id', 's_store_name', 's_street_type', 's_state', 's_county', 's_suite_number', 's_street_name', 's_gmt_offset', 's_street_number', 's_store_sk', 's_city'], 'promotion.csv': ['p_channel_event', 'p_channel_tv', 'p_channel_email', 'p_promo_sk', 'p_channel_dmail'], 'catalog_page.csv': ['cp_catalog_page_sk', 'cp_catalog_page_id'], 'web_page.csv': ['wp_char_count', 'wp_web_page_sk'], 'ship_mode.csv': ['sm_type', 'sm_ship_mode_sk', 'sm_carrier'], 'time_dim.csv': ['t_time', 't_minute', 't_time_sk', 't_hour', 't_meal_time'], 'household_demographics.csv': ['hd_dep_count', 'hd_demo_sk', 'hd_vehicle_count', 'hd_buy_potential', 'hd_income_band_sk'], 'reason.csv': ['r_reason_desc', 'r_reason_sk'], 'web_site.csv': ['web_name', 'web_company_name', 'web_site_id', 'web_site_sk'], 'income_band.csv': ['ib_income_band_sk', 'ib_lower_bound', 'ib_upper_bound'], 'call_center.csv': ['cc_call_center_sk', 'cc_name', 'cc_call_center_id', 'cc_county', 'cc_manager'], 'catalog_sales_dup_1_.csv': ['cs_sales_price', 'cs_ext_sales_price', 'cs_warehouse_sk', 'cs_ext_discount_amt', 'cs_net_profit', 'cs_bill_cdemo_sk', 'cs_promo_sk', 'cs_item_sk', 'cs_bill_hdemo_sk', 'cs_ship_customer_sk', 'cs_quantity', 'cs_wholesale_cost', 'cs_ext_ship_cost', 'cs_coupon_amt', 'cs_ext_list_price', 'cs_ship_cdemo_sk', 'cs_sold_time_sk', 'cs_net_paid', 'cs_ship_addr_sk', 'cs_ext_wholesale_cost', 'cs_list_price', 'cs_order_number', 'cs_catalog_page_sk', 'cs_bill_customer_sk', 'cs_sold_date_sk', 'cs_ship_date_sk', 'cs_ship_mode_sk', 'cs_call_center_sk', 'cs_bill_addr_sk'], 'catalog_sales_dup_2_.csv': ['cs_sales_price', 'cs_ext_sales_price', 'cs_warehouse_sk', 'cs_ext_discount_amt', 'cs_net_profit', 'cs_bill_cdemo_sk', 'cs_promo_sk', 'cs_item_sk', 'cs_bill_hdemo_sk', 'cs_ship_customer_sk', 'cs_quantity', 'cs_wholesale_cost', 'cs_ext_ship_cost', 'cs_coupon_amt', 'cs_ext_list_price', 'cs_ship_cdemo_sk', 'cs_sold_time_sk', 'cs_net_paid', 'cs_ship_addr_sk', 'cs_ext_wholesale_cost', 'cs_list_price', 'cs_order_number', 'cs_catalog_page_sk', 'cs_bill_customer_sk', 'cs_sold_date_sk', 'cs_ship_date_sk', 'cs_ship_mode_sk', 'cs_call_center_sk', 'cs_bill_addr_sk'], 'customer_address_dup_1_.csv': ['ca_street_name', 'ca_gmt_offset', 'ca_zip', 'ca_location_type', 'ca_suite_number', 'ca_country', 'ca_street_number', 'ca_city', 'ca_county', 'ca_address_sk', 'ca_street_type', 'ca_state'], 'customer_address_dup_2_.csv': ['ca_street_name', 'ca_gmt_offset', 'ca_zip', 'ca_location_type', 'ca_suite_number', 'ca_country', 'ca_street_number', 'ca_city', 'ca_county', 'ca_address_sk', 'ca_street_type', 'ca_state'], 'customer_dup_1_.csv': ['c_first_sales_date_sk', 'c_current_cdemo_sk', 'c_last_review_date_sk', 'c_current_hdemo_sk', 'c_customer_id', 'c_birth_month', 'c_birth_year', 'c_first_shipto_date_sk', 'c_birth_day', 'c_salutation', 'c_login', 'c_email_address', 'c_customer_sk', 'c_current_addr_sk', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_country', 'c_first_name'], 'call_center_dup_1_.csv': ['cc_call_center_sk', 'cc_name', 'cc_call_center_id', 'cc_county', 'cc_manager'], 'catalog_page_dup_1_.csv': ['cp_catalog_page_sk', 'cp_catalog_page_id'], 'customer_address_dup_3_.csv': ['ca_street_name', 'ca_gmt_offset', 'ca_zip', 'ca_location_type', 'ca_suite_number', 'ca_country', 'ca_street_number', 'ca_city', 'ca_county', 'ca_address_sk', 'ca_street_type', 'ca_state'], 'customer_demographics_dup_1_.csv': ['cd_dep_count', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_employed_count', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_demo_sk', 'cd_dep_college_count'], 'customer_dup_2_.csv': ['c_first_sales_date_sk', 'c_current_cdemo_sk', 'c_last_review_date_sk', 'c_current_hdemo_sk', 'c_customer_id', 'c_birth_month', 'c_birth_year', 'c_first_shipto_date_sk', 'c_birth_day', 'c_salutation', 'c_login', 'c_email_address', 'c_customer_sk', 'c_current_addr_sk', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_country', 'c_first_name'], 'date_dim_dup_1_.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'date_dim_dup_2_.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'customer_address_dup_4_.csv': ['ca_street_name', 'ca_gmt_offset', 'ca_zip', 'ca_location_type', 'ca_suite_number', 'ca_country', 'ca_street_number', 'ca_city', 'ca_county', 'ca_address_sk', 'ca_street_type', 'ca_state'], 'customer_demographics_dup_2_.csv': ['cd_dep_count', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_employed_count', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_demo_sk', 'cd_dep_college_count'], 'household_demographics_dup_1_.csv': ['hd_dep_count', 'hd_demo_sk', 'hd_vehicle_count', 'hd_buy_potential', 'hd_income_band_sk'], 'date_dim_dup_3_.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'date_dim_dup_4_.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'date_dim_dup_5_.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'date_dim_dup_6_.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'warehouse_dup_1_.csv': ['w_warehouse_name', 'w_warehouse_sq_ft', 'w_country', 'w_state', 'w_county', 'w_warehouse_sk', 'w_city'], 'customer_demographics_dup_3_.csv': ['cd_dep_count', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_employed_count', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_demo_sk', 'cd_dep_college_count'], 'customer_dup_3_.csv': ['c_first_sales_date_sk', 'c_current_cdemo_sk', 'c_last_review_date_sk', 'c_current_hdemo_sk', 'c_customer_id', 'c_birth_month', 'c_birth_year', 'c_first_shipto_date_sk', 'c_birth_day', 'c_salutation', 'c_login', 'c_email_address', 'c_customer_sk', 'c_current_addr_sk', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_country', 'c_first_name'], 'store_sales_dup_1_.csv': ['ss_cdemo_sk', 'ss_coupon_amt', 'ss_ext_list_price', 'ss_ext_sales_price', 'ss_item_sk', 'ss_store_sk', 'ss_ext_discount_amt', 'ss_sold_time_sk', 'ss_sold_date_sk', 'ss_ext_tax', 'ss_customer_sk', 'ss_net_profit', 'ss_ext_wholesale_cost', 'ss_wholesale_cost', 'ss_ticket_number', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_promo_sk', 'ss_list_price', 'ss_sales_price', 'ss_net_paid', 'ss_quantity'], 'date_dim_dup_7_.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'store_sales_dup_2_.csv': ['ss_cdemo_sk', 'ss_coupon_amt', 'ss_ext_list_price', 'ss_ext_sales_price', 'ss_item_sk', 'ss_store_sk', 'ss_ext_discount_amt', 'ss_sold_time_sk', 'ss_sold_date_sk', 'ss_ext_tax', 'ss_customer_sk', 'ss_net_profit', 'ss_ext_wholesale_cost', 'ss_wholesale_cost', 'ss_ticket_number', 'ss_hdemo_sk', 'ss_addr_sk', 'ss_promo_sk', 'ss_list_price', 'ss_sales_price', 'ss_net_paid', 'ss_quantity'], 'customer_address_dup_5_.csv': ['ca_street_name', 'ca_gmt_offset', 'ca_zip', 'ca_location_type', 'ca_suite_number', 'ca_country', 'ca_street_number', 'ca_city', 'ca_county', 'ca_address_sk', 'ca_street_type', 'ca_state'], 'customer_demographics_dup_4_.csv': ['cd_dep_count', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_employed_count', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_demo_sk', 'cd_dep_college_count'], 'customer_dup_4_.csv': ['c_first_sales_date_sk', 'c_current_cdemo_sk', 'c_last_review_date_sk', 'c_current_hdemo_sk', 'c_customer_id', 'c_birth_month', 'c_birth_year', 'c_first_shipto_date_sk', 'c_birth_day', 'c_salutation', 'c_login', 'c_email_address', 'c_customer_sk', 'c_current_addr_sk', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_country', 'c_first_name'], 'household_demographics_dup_2_.csv': ['hd_dep_count', 'hd_demo_sk', 'hd_vehicle_count', 'hd_buy_potential', 'hd_income_band_sk'], 'promotion_dup_1_.csv': ['p_channel_event', 'p_channel_tv', 'p_channel_email', 'p_promo_sk', 'p_channel_dmail'], 'date_dim_dup_8_.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'time_dim_dup_1_.csv': ['t_time', 't_minute', 't_time_sk', 't_hour', 't_meal_time'], 'store_dup_1_.csv': ['s_market_id', 's_zip', 's_company_name', 's_number_employees', 's_company_id', 's_store_id', 's_store_name', 's_street_type', 's_state', 's_county', 's_suite_number', 's_street_name', 's_gmt_offset', 's_street_number', 's_store_sk', 's_city'], 'web_sales_dup_1_.csv': ['ws_web_page_sk', 'ws_bill_addr_sk', 'ws_item_sk', 'ws_ext_discount_amt', 'ws_ext_sales_price', 'ws_net_paid', 'ws_ext_wholesale_cost', 'ws_wholesale_cost', 'ws_ship_hdemo_sk', 'ws_ship_customer_sk', 'ws_ship_addr_sk', 'ws_bill_customer_sk', 'ws_ship_date_sk', 'ws_net_profit', 'ws_sold_time_sk', 'ws_bill_cdemo_sk', 'ws_warehouse_sk', 'ws_sales_price', 'ws_sold_date_sk', 'ws_order_number', 'ws_promo_sk', 'ws_list_price', 'ws_web_site_sk', 'ws_ext_ship_cost', 'ws_ship_cdemo_sk', 'ws_ship_mode_sk', 'ws_quantity', 'ws_ext_list_price'], 'web_sales_dup_2_.csv': ['ws_web_page_sk', 'ws_bill_addr_sk', 'ws_item_sk', 'ws_ext_discount_amt', 'ws_ext_sales_price', 'ws_net_paid', 'ws_ext_wholesale_cost', 'ws_wholesale_cost', 'ws_ship_hdemo_sk', 'ws_ship_customer_sk', 'ws_ship_addr_sk', 'ws_bill_customer_sk', 'ws_ship_date_sk', 'ws_net_profit', 'ws_sold_time_sk', 'ws_bill_cdemo_sk', 'ws_warehouse_sk', 'ws_sales_price', 'ws_sold_date_sk', 'ws_order_number', 'ws_promo_sk', 'ws_list_price', 'ws_web_site_sk', 'ws_ext_ship_cost', 'ws_ship_cdemo_sk', 'ws_ship_mode_sk', 'ws_quantity', 'ws_ext_list_price'], 'reason_dup_1_.csv': ['r_reason_desc', 'r_reason_sk'], 'customer_address_dup_6_.csv': ['ca_street_name', 'ca_gmt_offset', 'ca_zip', 'ca_location_type', 'ca_suite_number', 'ca_country', 'ca_street_number', 'ca_city', 'ca_county', 'ca_address_sk', 'ca_street_type', 'ca_state'], 'customer_demographics_dup_5_.csv': ['cd_dep_count', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_employed_count', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_demo_sk', 'cd_dep_college_count'], 'date_dim_dup_9_.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'customer_address_dup_7_.csv': ['ca_street_name', 'ca_gmt_offset', 'ca_zip', 'ca_location_type', 'ca_suite_number', 'ca_country', 'ca_street_number', 'ca_city', 'ca_county', 'ca_address_sk', 'ca_street_type', 'ca_state'], 'customer_demographics_dup_6_.csv': ['cd_dep_count', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_employed_count', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_demo_sk', 'cd_dep_college_count'], 'customer_dup_5_.csv': ['c_first_sales_date_sk', 'c_current_cdemo_sk', 'c_last_review_date_sk', 'c_current_hdemo_sk', 'c_customer_id', 'c_birth_month', 'c_birth_year', 'c_first_shipto_date_sk', 'c_birth_day', 'c_salutation', 'c_login', 'c_email_address', 'c_customer_sk', 'c_current_addr_sk', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_country', 'c_first_name'], 'customer_address_dup_8_.csv': ['ca_street_name', 'ca_gmt_offset', 'ca_zip', 'ca_location_type', 'ca_suite_number', 'ca_country', 'ca_street_number', 'ca_city', 'ca_county', 'ca_address_sk', 'ca_street_type', 'ca_state'], 'customer_demographics_dup_7_.csv': ['cd_dep_count', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_employed_count', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_demo_sk', 'cd_dep_college_count'], 'customer_dup_6_.csv': ['c_first_sales_date_sk', 'c_current_cdemo_sk', 'c_last_review_date_sk', 'c_current_hdemo_sk', 'c_customer_id', 'c_birth_month', 'c_birth_year', 'c_first_shipto_date_sk', 'c_birth_day', 'c_salutation', 'c_login', 'c_email_address', 'c_customer_sk', 'c_current_addr_sk', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_country', 'c_first_name'], 'promotion_dup_2_.csv': ['p_channel_event', 'p_channel_tv', 'p_channel_email', 'p_promo_sk', 'p_channel_dmail'], 'customer_address_dup_9_.csv': ['ca_street_name', 'ca_gmt_offset', 'ca_zip', 'ca_location_type', 'ca_suite_number', 'ca_country', 'ca_street_number', 'ca_city', 'ca_county', 'ca_address_sk', 'ca_street_type', 'ca_state'], 'customer_demographics_dup_8_.csv': ['cd_dep_count', 'cd_purchase_estimate', 'cd_credit_rating', 'cd_dep_employed_count', 'cd_gender', 'cd_marital_status', 'cd_education_status', 'cd_demo_sk', 'cd_dep_college_count'], 'customer_dup_7_.csv': ['c_first_sales_date_sk', 'c_current_cdemo_sk', 'c_last_review_date_sk', 'c_current_hdemo_sk', 'c_customer_id', 'c_birth_month', 'c_birth_year', 'c_first_shipto_date_sk', 'c_birth_day', 'c_salutation', 'c_login', 'c_email_address', 'c_customer_sk', 'c_current_addr_sk', 'c_last_name', 'c_preferred_cust_flag', 'c_birth_country', 'c_first_name'], 'date_dim_dup_10_.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'household_demographics_dup_3_.csv': ['hd_dep_count', 'hd_demo_sk', 'hd_vehicle_count', 'hd_buy_potential', 'hd_income_band_sk'], 'ship_mode_dup_1_.csv': ['sm_type', 'sm_ship_mode_sk', 'sm_carrier'], 'date_dim_dup_11_.csv': ['d_date', 'd_moy', 'd_dow', 'd_dom', 'd_qoy', 'd_quarter_name', 'd_day_name', 'd_date_sk', 'd_month_seq', 'd_week_seq', 'd_year'], 'time_dim_dup_2_.csv': ['t_time', 't_minute', 't_time_sk', 't_hour', 't_meal_time'], 'warehouse_dup_2_.csv': ['w_warehouse_name', 'w_warehouse_sq_ft', 'w_country', 'w_state', 'w_county', 'w_warehouse_sk', 'w_city'], 'web_page_dup_1_.csv': ['wp_char_count', 'wp_web_page_sk']})
    TPCDS_BENCHMARK_PRED_COLS = collections.defaultdict(list,{
        'catalog_returns.csv': ['cr_return_amount',  'cr_order_number',  'cr_returning_addr_sk',  'cr_call_center_sk',  'cr_net_loss',  'cr_reversed_charge',  'cr_store_credit',  'cr_catalog_page_sk',  'cr_return_quantity',  'cr_returned_date_sk',  'cr_returning_customer_sk',  'cr_return_amt_inc_tax',  'cr_refunded_addr_sk',  'cr_item_sk',  'cr_refunded_cash'],
        'customer.csv': ['c_first_sales_date_sk',  'c_current_cdemo_sk',  'c_last_review_date_sk',  'c_current_hdemo_sk',  'c_customer_id',  'c_birth_month',  'c_birth_year',  'c_first_shipto_date_sk',  'c_birth_day',  'c_salutation',  'c_login',  'c_email_address',  'c_customer_sk',  'c_current_addr_sk',  'c_last_name',  'c_preferred_cust_flag',  'c_birth_country',  'c_first_name'],
        'customer_address.csv': ['ca_street_name',  'ca_gmt_offset',  'ca_zip',  'ca_location_type',  'ca_suite_number',  'ca_country',  'ca_street_number',  'ca_city',  'ca_county',  'ca_address_sk',  'ca_street_type',  'ca_state'],
        'catalog_sales.csv': ['cs_sales_price',  'cs_ext_sales_price',  'cs_warehouse_sk',  'cs_ext_discount_amt',  'cs_net_profit',  'cs_bill_cdemo_sk',  'cs_promo_sk',  'cs_item_sk',  'cs_bill_hdemo_sk',  'cs_ship_customer_sk',  'cs_quantity',  'cs_wholesale_cost',  'cs_ext_ship_cost',  'cs_coupon_amt',  'cs_ext_list_price',  'cs_ship_cdemo_sk',  'cs_sold_time_sk',  'cs_net_paid',  'cs_ship_addr_sk',  'cs_ext_wholesale_cost',  'cs_list_price',  'cs_order_number',  'cs_catalog_page_sk',  'cs_bill_customer_sk',  'cs_sold_date_sk',  'cs_ship_date_sk',  'cs_ship_mode_sk',  'cs_call_center_sk',  'cs_bill_addr_sk'],
        'warehouse.csv': ['w_warehouse_name',  'w_warehouse_sq_ft',  'w_country',  'w_state',  'w_county',  'w_warehouse_sk',  'w_city'],
        'store_returns.csv': ['sr_reason_sk',  'sr_returned_date_sk',  'sr_return_amt',  'sr_item_sk',  'sr_customer_sk',  'sr_return_quantity',  'sr_cdemo_sk',  'sr_store_sk',  'sr_ticket_number',  'sr_net_loss'],
        'customer_demographics.csv': ['cd_dep_count',  'cd_purchase_estimate',  'cd_credit_rating',  'cd_dep_employed_count',  'cd_gender',  'cd_marital_status',  'cd_education_status',  'cd_demo_sk',  'cd_dep_college_count'],
        'web_returns.csv': ['wr_fee',  'wr_returned_date_sk',  'wr_reason_sk',  'wr_web_page_sk',  'wr_item_sk',  'wr_returning_cdemo_sk',  'wr_return_quantity',  'wr_return_amt',  'wr_refunded_addr_sk',  'wr_order_number',  'wr_returning_addr_sk',  'wr_net_loss',  'wr_refunded_cash',  'wr_refunded_cdemo_sk',  'wr_returning_customer_sk'],
        'inventory.csv': ['inv_quantity_on_hand',  'inv_item_sk',  'inv_warehouse_sk',  'inv_date_sk'],
        'web_sales.csv': ['ws_web_page_sk',  'ws_bill_addr_sk',  'ws_item_sk',  'ws_ext_discount_amt',  'ws_ext_sales_price',  'ws_net_paid',  'ws_ext_wholesale_cost',  'ws_wholesale_cost',  'ws_ship_hdemo_sk',  'ws_ship_customer_sk',  'ws_ship_addr_sk',  'ws_bill_customer_sk',  'ws_ship_date_sk',  'ws_net_profit',  'ws_sold_time_sk',  'ws_bill_cdemo_sk',  'ws_warehouse_sk',  'ws_sales_price',  'ws_sold_date_sk',  'ws_order_number',  'ws_promo_sk',  'ws_list_price',  'ws_web_site_sk',  'ws_ext_ship_cost',  'ws_ship_cdemo_sk',  'ws_ship_mode_sk',  'ws_quantity',  'ws_ext_list_price'],
        'date_dim.csv': ['d_date',  'd_moy',  'd_dow',  'd_dom',  'd_qoy',  'd_quarter_name',  'd_day_name',  'd_date_sk',  'd_month_seq',  'd_week_seq',  'd_year'],
        'store_sales.csv': ['ss_cdemo_sk',  'ss_coupon_amt',  'ss_ext_list_price',  'ss_ext_sales_price',  'ss_item_sk',  'ss_store_sk',  'ss_ext_discount_amt',  'ss_sold_time_sk',  'ss_sold_date_sk',  'ss_ext_tax',  'ss_customer_sk',  'ss_net_profit',  'ss_ext_wholesale_cost',  'ss_wholesale_cost',  'ss_ticket_number',  'ss_hdemo_sk',  'ss_addr_sk',  'ss_promo_sk',  'ss_list_price',  'ss_sales_price',  'ss_net_paid',  'ss_quantity'],
        'item.csv': ['i_manager_id',  'i_wholesale_cost',  'i_brand',  'i_manufact',  'i_size',  'i_category',  'i_item_sk',  'i_class',  'i_current_price',  'i_brand_id',  'i_item_desc',  'i_category_id',  'i_item_id',  'i_product_name',  'i_color',  'i_manufact_id',  'i_units',  'i_class_id'],
        'store.csv': ['s_market_id',  's_zip',  's_company_name',  's_number_employees',  's_company_id',  's_store_id',  's_store_name',  's_street_type',  's_state',  's_county',  's_suite_number',  's_street_name',  's_gmt_offset',  's_street_number',  's_store_sk',  's_city'],
        'promotion.csv': ['p_channel_event',  'p_channel_tv',  'p_channel_email',  'p_promo_sk',  'p_channel_dmail'],
        'catalog_page.csv': ['cp_catalog_page_sk', 'cp_catalog_page_id'],
        'web_page.csv': ['wp_char_count', 'wp_web_page_sk'],
        'ship_mode.csv': ['sm_type', 'sm_ship_mode_sk', 'sm_carrier'],
        'time_dim.csv': ['t_time', 't_minute', 't_time_sk', 't_hour', 't_meal_time'],
        'household_demographics.csv': ['hd_dep_count',  'hd_demo_sk',  'hd_vehicle_count',  'hd_buy_potential',  'hd_income_band_sk'],
        'reason.csv': ['r_reason_desc', 'r_reason_sk'],
        'web_site.csv': ['web_name', 'web_company_name', 'web_site_id', 'web_site_sk'],
        'income_band.csv': ['ib_income_band_sk', 'ib_lower_bound', 'ib_upper_bound'],
        'call_center.csv': ['cc_call_center_sk',  'cc_name',  'cc_call_center_id',  'cc_county',  'cc_manager']
    })

    CSV_FILES = ['call_center.csv', 'catalog_page.csv', 'catalog_returns.csv', 'catalog_sales.csv',
                 'customer.csv', 'household_demographics.csv', 'inventory.csv', 'promotion.csv',
                 'store.csv', 'store_returns.csv', 'store_sales.csv', 'web_page.csv',
                 'web_returns.csv', 'web_sales.csv', 'web_site.csv', 'date_dim.csv',
                 'item.csv', 'reason.csv', 'customer_address.csv', 'customer_demographics.csv',
                 'time_dim.csv', 'ship_mode.csv', 'warehouse.csv', 'income_band.csv']

    TRUE_FULL_OUTER_CARDINALITY = {

        # ('call_center', 'catalog_page', 'catalog_returns', 'catalog_sales', 'customer', 'customer_address', 'customer_demographics', 'date_dim', 'household_demographics', 'income_band', 'inventory', 'item', 'promotion', 'reason', 'ship_mode', 'store', 'store_returns', 'store_sales', 'time_dim', 'warehouse', 'web_page', 'web_returns', 'web_sales', 'web_site'): 6995040586160183733,

    }

    @staticmethod
    def GetFullOuterCardinalityOrFail(join_tables):
        key = tuple(sorted(join_tables))
        return TPCDSBenchmark.TRUE_FULL_OUTER_CARDINALITY[key]

def LoadTPCDS(table=None,
              data_dir='./datasets/tpcds_2_13_0/',
              try_load_parsed=True,
              use_cols='tpcds-db'):
    def TryLoad(table_name, filepath, use_cols, **kwargs):
        """Try load from previously parsed (table, columns)."""
        if use_cols:
            cols_str = '-'.join(use_cols)
            parsed_path = filepath[:-4] + '.{}.table'.format(cols_str)
        else:
            parsed_path = filepath[:-4] + '.table'

        parsed_path = filename_encoder(parsed_path)
        try_load_parsed = False

        if try_load_parsed:
            if os.path.exists(parsed_path):
                arr = np.load(parsed_path, allow_pickle=True)
                print('Loaded parsed Table from', parsed_path)
                table = arr.item()
                print(table)
                return table
        table = CsvTable(
            table_name,
            filepath,
            cols=use_cols,
            **kwargs,
        )
        if try_load_parsed:
            np.save(open(parsed_path, 'wb'), table)
            print('Saved parsed Table to', parsed_path)
        return table

    def get_use_cols(filepath):
        print(f'get use_cols {use_cols}\n\n')
        if use_cols == 'tpcds-db':
            return TPCDSBenchmark.TPCDS_DB_PRED_COLS.get(filepath, None)
        if use_cols == 'tpcds-db-dup':
            return TPCDSBenchmark.TPCDS_DB_DUP_PRED_COLS.get(filepath, None)
        if use_cols == 'tpcds-benchmark':
            return TPCDSBenchmark.TPCDS_BENCHMARK_PRED_COLS.get(filepath, None)
        if use_cols == 'tpcds-benchmark-dup':
            return TPCDSBenchmark.TPCDS_BENCHMARK_DUP_PRED_COLS.get(filepath, None)
        return None  # Load all.

    if table:
        filepath = table + '.csv'
        table = TryLoad(
            table,
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
        )
        return table

    tables = {}

    file_path_list = list()
    if use_cols == 'tpcds-db':
        file_path_list = TPCDSBenchmark.TPCDS_DB_PRED_COLS
    elif use_cols == 'tpcds-db-dup':
        file_path_list = TPCDSBenchmark.TPCDS_DB_DUP_PRED_COLS
    elif use_cols == 'tpcds-benchmark':
        file_path_list = TPCDSBenchmark.TPCDS_BENCHMARK_PRED_COLS
    elif use_cols == 'tpcds-benchmark-dup':
        file_path_list = TPCDSBenchmark.TPCDS_BENCHMARK_DUP_PRED_COLS
    else: return None
    for filepath in file_path_list:
        tables[filepath[0:-4]] = TryLoad(
            filepath[0:-4],
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
        )

    return tables


def LoadDataset(dataset, table, use_cols, data_dir,try_load_parsed=True):
    if dataset == 'imdb':
        return LoadImdb(table, data_dir=data_dir,try_load_parsed=try_load_parsed, use_cols=use_cols)
    if dataset == 'tpcds':
        return LoadTPCDS(table,data_dir=data_dir, try_load_parsed=try_load_parsed, use_cols=use_cols)
    if dataset == 'toy':
        return LoadTOYTest(table,data_dir=data_dir,try_load_parsed=try_load_parsed,use_cols=use_cols)
    if dataset == 'synthetic':
        return LoadSYN(table,data_dir=data_dir,try_load_parsed=try_load_parsed,use_cols=use_cols)


def filename_encoder(file_path, map_path='../tools/encode_map.pkl'):
    def encoding(txt, mapper, delimeter='-'):
        tokens = txt.split(delimeter)
        result = []
        for token in tokens:
            if token not in mapper.keys():
                result.append(token)

            elif mapper[token]:
                result.append(mapper[token])
            else:
                result.append(token)
        return '-'.join(result)

    name_dict = pickle.load(open(map_path, 'rb'))
    path = '/'.join(file_path.split('/')[:-1])
    file = file_path.split('/')[-1]
    tokens = file.split('.')[:-1]

    filename = ''
    for token in tokens:
        txt = encoding(token, name_dict)
        if txt is None:
            return file_path
        filename += txt + '.'

    filename += file.split('.')[-1]

    return os.path.join(path, filename)

def get_cardinality(dataset, join_tables):
    if dataset == 'imdb':
        return JoinOrderBenchmark.GetFullOuterCardinalityOrFail(
            join_tables)
    if dataset == 'tpcds':
        return TPCDSBenchmark.GetFullOuterCardinalityOrFail(
            join_tables)
    if dataset == 'toy':
        return TOYTESTBenchmark.GetFullOuterCardinalityOrFail(join_tables)
    if dataset == 'synthetic':
        return SyntheticDataset.GetFullOuterCardinalityOrFail(join_tables)

def get_use_column(dataset, table, usecols):
    if dataset == 'imdb':
        if usecols == 'simple':
            return JoinOrderBenchmark.BASE_TABLE_PRED_COLS[f"{table}.csv"]
        elif usecols == 'content':
            return JoinOrderBenchmark.ContentColumns()[f"{table}.csv"]
        elif usecols == 'multi':
            return JoinOrderBenchmark.JOB_M_PRED_COLS[f"{table}.csv"]
        elif usecols == 'toy':
            return JoinOrderBenchmark.JOB_TOY_PRED_COLS[f"{table}.csv"]
        elif usecols =='imdb-db':
            return JoinOrderBenchmark.IMDB_DB_PRED_COLS[f"{table}.csv"]
        elif usecols =='imdb-full-dup':
            return JoinOrderBenchmark.IMDB_FULL_DUP_PRED_COLS[f"{table}.csv"]
        elif usecols == 'union':
            return JoinOrderBenchmark.JOB_UNION_PRED_COLS[f"{table}.csv"]
        elif usecols == 'deepdb':
            return JoinOrderBenchmark.JOB_DEEPDB_PRED_COLS[f"{table}.csv"]
        elif usecols == 'imdb-full-deepdb':
            return JoinOrderBenchmark.IMDB_DEEPDB_PRED_COLS[f"{table}.csv"]
        elif usecols =='job':
            return JoinOrderBenchmark.JOB_ORIGINAL_PRED_COLS[f"{table}.csv"]
        elif usecols == 'original_dup':
            return JoinOrderBenchmark.JOB_ORIGINAL_DUP_PRED[f"{table}.csv"]
        elif usecols =='uae-light':
            return JoinOrderBenchmark.UAE_PRED_COLS[f"{table}.csv"]
        return None
    if dataset =='tpcds':
        if usecols == 'tpcds-db':
            return TPCDSBenchmark.TPCDS_DB_PRED_COLS[f"{table}.csv"]
        if usecols == 'tpcds-db-dup':
            return TPCDSBenchmark.TPCDS_DB_DUP_PRED_COLS[f"{table}.csv"]
        if usecols == 'tpcds-benchmark':
            return TPCDSBenchmark.TPCDS_BENCHMARK_PRED_COLS[f"{table}.csv"]
        if usecols == 'tpcds-benchmark-dup':
            return TPCDSBenchmark.TPCDS_BENCHMARK_DUP_PRED_COLS[f"{table}.csv"]
        if usecols == 'tpcds-toy':
            return TPCDSBenchmark.TPCDS_TOY_PRED_COLS[f"{table}.csv"]
    if dataset == 'synthetic':
        if usecols == 'multi':
            return SyntheticDataset.SYN_MULTI_PRED_COLS[f"{table}.csv"]
        if usecols == 'multi_rev':
            return SyntheticDataset.SYN_MULTI_REV_PRED_COLS[f"{table}.csv"]
        if usecols == 'single':
            return SyntheticSingleDataset.SYN_SINGLE_PRED_COLS[f"{table}.csv"]




class TOYTESTBenchmark(object) :
    TOY_TEST_PRED_COLS = collections.defaultdict(list, {
        'A.csv' : ['x'], 'B.csv': ['x','y'], 'C.csv' : ['y']} )
    CSV_FILES = ['A.csv','B.csv','C.csv']
    TRUE_FULL_OUTER_CARDINALITY = { ('A','B','C'):5}

    @staticmethod
    def GetFullOuterCardinalityOrFail(join_tables):
        key = tuple(sorted(join_tables))
        return TOYTESTBenchmark.TRUE_FULL_OUTER_CARDINALITY[key]

def LoadTOYTest(table=None,
                data_dir='./datasets/test/',
                try_load_parsed=True,
                use_cols='toy_test_col'):
    def TryLoad(table_name, filepath, use_cols, **kwargs):
        """Try load from previously parsed (table, columns)."""
        if use_cols:
            cols_str = '-'.join(use_cols)
            parsed_path = filepath[:-4] + '.{}.table'.format(cols_str)
        else:
            parsed_path = filepath[:-4] + '.table'

        if try_load_parsed:
            if os.path.exists(parsed_path):
                arr = np.load(parsed_path, allow_pickle=True)
                print('Loaded parsed Table from', parsed_path)
                table = arr.item()
                print(table)
                return table
        table = CsvTable(
            table_name,
            filepath,
            cols=use_cols,
            **kwargs,
        )
        if try_load_parsed:
            np.save(open(parsed_path, 'wb'), table)
            print('Saved parsed Table to', parsed_path)
        return table

    def get_use_cols(filepath):
        if use_cols == 'toy_test_col':
            return TOYTESTBenchmark.TOY_TEST_PRED_COLS.get(filepath, None)
        return None  # Load all.

    if table:
        filepath = table + '.csv'
        table = TryLoad(
            table,
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
        )
        return table

    tables = {}
    for filepath in TOYTESTBenchmark.TOY_TEST_PRED_COLS:
        tables[filepath[0:-4]] = TryLoad(
            filepath[0:-4],
            data_dir + filepath,
            use_cols=get_use_cols(filepath),
            type_casts={},
        )

    return tables


