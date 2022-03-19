"""
query 를 입력 받았을 때, 그 query 의 cardinality 를 구하는 프로그램.
목적 : 
- join 이 많이 포함된 쿼리에 대해 빠르게 cardinality 를 구하기 위함. 
- neurocard 의 sampler를 생성하는 과정에서 jct 를 linear time 으로 생성할 수 있음. 
- jct 에서 root table 의 weight sum 이 full outer join 의 cardinality 가 됨. 
- 이를 활용, 쿼리에 나오는 테이블들에 filter(predicate) 를 적용한 후 jct 생성, cardinality 를 구한다.  


입력
- 쿼리 파일 형식 (테이블|조인절|predicate)
- dataset csv 파일
- 파일 명 [workload].csv
출력
- 쿼리 파일 (테이블|조인절|predicate|card)
- 파일명 [workload]_[# of rows]_[date].out

pseudo code
- 쿼리 한줄 식 read
- 쿼리에 사용된 테이블 load (csv 파일 -> pd.dataframe)
- load 시 predicate 에 해당하는 row 만 load (masking)
- load 된 테이블 join key 로 bct, jct 생성
- join root 테이블의  jct 의 weight sum 으로 cardinality 구함.
"""


import utils
import datasets
import common
import pandas as pd
import numpy as np
import join_utils
import collections
import os
import argparse
import time
import datetime
import ray
import ast
import csv

parser = argparse.ArgumentParser()

parser.add_argument('--input',type=str,required=True,help='input file paths')
parser.add_argument('--output',default=None,type=str,required=False,help='output file path')
parser.add_argument('--dataset',default='tpcds',type=str,required=False,help='dataset')
parser.add_argument('--data_dir',default='datasets/tpcds_2_13_0',type=str,required=False,help='data directory')
parser.add_argument('--join_root',default='item',type=str,required=False,help='data directory')
parser.add_argument('--sep',default='|',type=str,required=False,help='seperator')


args = parser.parse_args()


NULL = -1


TPCDS_ALIAS_DICT= {'ss': 'store_sales','sr': 'store_returns','cs': 'catalog_sales','cr': 'catalog_returns','ws': 'web_sales','wr': 'web_returns','inv': 'inventory','s': 'store','cc': 'call_center','cp': 'catalog_page','web': 'web_site','wp': 'web_page','w': 'warehouse','c': 'customer','ca': 'customer_address','cd': 'customer_demographics','d': 'date_dim','hd': 'household_demographics','i': 'item','ib': 'income_band','p': 'promotion','r': 'reason','sm': 'ship_mode','t': 'time_dim'}
JOB_ALIAS_DICT = {'ci': 'cast_info', 'ct': 'company_type', 'mc': 'movie_companies', 't': 'title', 'cn': 'company_name', 'k': 'keyword', 'mi_idx': 'movie_info_idx', 'it': 'info_type', 'mi': 'movie_info', 'mk': 'movie_keyword' }

TPCDS_TABLES = ['store_sales','store_returns','catalog_sales','catalog_returns','web_sales','web_returns','inventory','store','call_center','catalog_page','web_site','web_page','warehouse','customer','customer_address','customer_demographics','date_dim','household_demographics','item','income_band','promotion','reason','ship_mode','time_dim']
JOB_TABLES = ['title', 'aka_title', 'cast_info', 'complete_cast', 'movie_companies', 'movie_info', 'movie_info_idx', 'movie_keyword', 'movie_link', 'kind_type', 'comp_cast_type', 'company_name', 'company_type', 'info_type', 'keyword', 'link_type']

JoinSpec = collections.namedtuple("JoinSpec", [
    "join_tables", "join_keys", "join_clauses", "join_graph", "join_tree",
    "join_root", "join_how", "join_name"
])

def get_join_spec(config):
    join_clauses = config["join_clauses"]
    if join_clauses is None:
        join_clauses = join_utils._infer_join_clauses(config["join_tables"],
                                           config["join_keys"],
                                           config["join_root"])
    g, dg = join_utils._make_join_graph(join_clauses, config["join_root"])
    return JoinSpec(
        join_tables=config["join_tables"],
        join_keys=config["join_keys"],
        join_clauses=join_clauses,
        join_graph=g,
        join_tree=dg,
        join_root=config["join_root"],
        join_how=config["join_how"],
        join_name="{}".format(config.get("join_name")),
    )



def tablePredDict(query_pred,query_tables) :
    table_pred_dict = dict()
    for t in query_tables :
        table_pred_dict[t] = list()
    for t,v in query_pred.items() :
        cols = v['cols']
        ops = v['ops']
        vals = v['vals']

        for pred_tup in zip(cols,ops,vals) :
            table_pred_dict[t].append(pred_tup)
            
    return table_pred_dict


def getJoinClauses(queries_csv,sep='|') :
    join_clauses = list()
    with open(queries_csv,'r') as file : 
        lines = file.readlines()
        for line in lines : 
            join_clause_txt = line.split(sep)[1]
            join_clause = join_clause_txt.split(',')
            join_clauses.append(join_clause)
    return join_clauses


def get_join_count_tables(loaded_tables,join_spec,pred_dict):
    t1 = time.time()
    base_count_tables_dict = {
        # +@ pass parameter
        table: get_base_count_table.remote(loaded_tables[table], predicates = pred_dict[table], keys=list(keys))
        for table, keys in join_spec.join_keys.items()
    }
    t2 = time.time()
    print(f"\tbase count table build : {t2-t1:0.2f}")
    join_count_tables_dict = {}
    t1 = time.time()
    for i,table in enumerate(join_utils.get_bottom_up_table_ordering(join_spec) ):
        
        dependencies = list(join_spec.join_tree.neighbors(table))
        if len(dependencies) == 0:
            jct = get_first_jct.remote(join_spec.join_name, table,
                                       base_count_tables_dict[table])
        else:
            bct = base_count_tables_dict[table]
            dependency_jcts = [join_count_tables_dict[d] for d in dependencies]
            jct = get_jct.remote(table, bct, dependencies, dependency_jcts,
                                 join_spec)
        join_count_tables_dict[table] = ray.get(jct)
    t2 = time.time()
    print(f"\tjoin count table build : {t2 - t1:0.2f}")
    return base_count_tables_dict,join_count_tables_dict

@ray.remote
def get_first_jct(join_name, table, base_count_table):
    ret = base_count_table
    ret.columns = [f"{table}.{k}" for k in ret.columns]
    return ret

@ray.remote
def get_jct(table, bct, dependencies, dependency_jcts, join_spec):

    jct_columns = [f"{table}.{k}" for k in bct.columns]
    bct.columns = jct_columns
    keys = join_spec.join_keys[table]
    groupby_keys = [f"{table}.{k}" for k in keys]
    table_weight = f"{table}.weight"
    ret_keys = groupby_keys + [table_weight]
    ret = bct[ret_keys]

    for other, other_jct in zip(dependencies, dependency_jcts):
        join_keys = join_spec.join_graph[table][other]["join_keys"]
        table_key = f"{table}.{join_keys[table]}"
        other_key = f"{other}.{join_keys[other]}"
        other_weight = f"{other}.weight"
        ret = ret.merge(
            other_jct[[other_key, other_weight]],
            how=join_spec.join_how,
            left_on=table_key,
            right_on=other_key,
        )
        ret[table_weight] = np.nanprod(
            [ret[table_weight], ret[other_weight]], axis=0)
        ret = ret[ret_keys]
        ret = ret.fillna(NULL).groupby(groupby_keys).sum().reset_index()

    if join_spec.join_how == "outer":
        assert 0 <= len(ret) - len(bct) <= 1, (ret, bct)

    bct_sans_weight = bct.drop(table_weight, axis=1)
    ret = ret.merge(bct_sans_weight,
                    how="left",
                    left_on=groupby_keys,
                    right_on=groupby_keys)
    ret = ret[jct_columns]

    jct = ret.fillna(1).astype(np.int64, copy=False)
    return jct

@ray.remote
def get_base_count_table(loaded_table,predicates,keys):

    df = filtered_table(loaded_table,predicates=predicates)
    groupby_ss = df.groupby(keys).size()
    bct = groupby_ss.to_frame(name="weight").reset_index()
    for key in keys:
        kct = df.groupby(key).size().rename(f"{key}.cnt")
        bct = bct.merge(kct, how="left", left_on=key, right_index=True)
        
    return bct.astype(np.int64, copy=False)


def filtered_table(table,predicates) :
    df = table
    for c,o,v in predicates :
        col_type = df[c].dtype
        try :
            val = pd.Series.astype(pd.Series([v]),col_type).item()
        except :
            val =int(float(v))
        pred = get_pred(c,o,val)
        df = df.query(pred,engine='python')#.reset_index()
    return df


# def load_table(table, predicates, data_dir, dataset , **kwargs):
#
#     # kwargs.update({"usecols": None})
#
#     df = pd.read_csv(os.path.join(data_dir, f"{table}.csv"),
#                            escapechar="\\",
#                            low_memory=False,
#                            **kwargs)
#
#     for c,o,v in predicates :
#         col_type = df[c].dtype
#         try :
#             val = pd.Series.astype(pd.Series([v]),col_type).item()
#         except :
#             val =int(float(v))
#         pred = get_pred(c,o,val)
#         df = df.query(pred,engine='python')#.reset_index()
#     return df

def get_pred(c,o,val):
    assert o in ['=', '>=','<=','<','>'], f"OP {o} is not allowed"
    if o == '=' :
        o = '=='
    if type(val)==str: 
        val = f"""\"{val}\"""" 
    return f"{c} {o} {val}"


def get_lines(path) :
    with open(path,'r') as file :
        lines = file.readlines()
    return lines

def append_output_line(path,input_line,card,sep) :
    txt = input_line.replace('\n',sep)
    output_txt = f"{txt[:-2]}{card}\n"
    with open(path,'at') as writer : 
        writer.write(output_txt)

def FormattingQuery(csv_file,sep, use_alias_keys=False):

    queries = []
    with open(csv_file) as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=sep))
        for row in data_raw:
            reader = csv.reader(row)  # comma-separated
            table_dict = utils._get_table_dict(next(reader))
            join_dict = utils._get_join_dict(next(reader), table_dict, use_alias_keys)
            predicate_dict = utils._get_predicate_dict(next(reader), table_dict)
            queries.append((list(table_dict.values()), join_dict,
                            predicate_dict))

        return queries


def load_tables(tables,data_dir,**kwargs) :
    table_dict = dict()
    for table in tables :
        df = pd.read_csv(os.path.join(data_dir, f"{table}.csv"),
                               low_memory=False,
                               **kwargs)
        table_dict[table] = df
    return table_dict
if __name__ == "__main__":

    ray.init(ignore_reinit_error=True)


    input_file = args.input
    output_file = args.output
    dataset = args.dataset
    data_dir = args.data_dir
    
    sep = args.sep

    now = datetime.datetime.now().strftime('%m%d')

    input_lines = get_lines(input_file)

    if output_file is None :
        output_file = f"{input_file.replace('.csv','')}_{len(input_lines)}_{now}.out"

    join_dict = {'join_how': 'inner',
              'join_name': f"{dataset}-card"}

    queries_job_format = FormattingQuery(input_file,sep=sep)
    join_clauses = getJoinClauses(input_file,sep=sep)

    if dataset == 'imdb' :
        table_list = JOB_TABLES
        alias_dict = JOB_ALIAS_DICT
    if dataset == 'tpcds' :
        table_list = TPCDS_TABLES
        alias_dict = TPCDS_ALIAS_DICT
    assert table_list is not None

    t1 = time.time()
    loaded_tables = load_tables(table_list,data_dir=data_dir)
    t2 = time.time()

    print(f"{dataset} Table Loading Time  : {t2-t1:0.2f} \n load : {list(loaded_tables.keys())} from {data_dir}")


    assert len(queries_job_format) == len(join_clauses)

    for i,(query_tables, join_keys, query_pred) in enumerate(queries_job_format): 
        join_clause = join_clauses[i]
        join_dict['join_tables'] = query_tables
        join_dict['join_keys'] = join_keys
        join_dict['join_clauses'] = join_clause
        join_dict["join_root"] = query_tables[0]
        join_root = query_tables[0]
        
        
        pred_dict = tablePredDict(query_pred,query_tables)  # table : (column, op, val) format
        
        if len(query_tables) > 1: 

            join_spec = get_join_spec(join_dict)
           

            t1 = time.time()
            bct,jct = get_join_count_tables(loaded_tables=loaded_tables,join_spec=join_spec,pred_dict =pred_dict )
            t2 = time.time()

            card = jct[join_root][f"{join_root}.weight"].sum()

            total_size = 0
            for _, df in bct.items(): total_size += ray.get(df).size
            print(f"Query [{i}] - card : {card} \t dur : {t2-t1:0.2f}s \t size :{total_size}")
        else: 
            table = query_tables[0]
            t1 = time.time()
            df = filtered_table(loaded_tables[table], predicates = pred_dict[table])
            t2 = time.time()
            card = len(df)
            print(f"Query [{i}] - card : {card} \t dur : {t2-t1:0.2f}s")
            
            
        append_output_line(output_file,input_lines[i],card,sep)