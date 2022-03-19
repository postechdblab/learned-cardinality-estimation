"""Utility functions."""

import ast
from collections import defaultdict
import csv


def _get_table_dict(tables):
    table_dict = {}
    for t in tables:
        split = t.split(' ')
        if len(split) > 1:
            # Alias -> full table name.
            table_dict[split[1]] = split[0]
        else:
            # Just full table name.
            table_dict[split[0]] = split[0]
    return table_dict


def _get_join_dict(joins, table_dict, use_alias_keys):
    join_dict = defaultdict(set)
    for j in joins:
        ops = j.split('=')
        op1 = ops[0].split('.')
        op2 = ops[1].split('.')
        t1, k1 = op1[0], op1[1]
        t2, k2 = op2[0], op2[1]
        if not use_alias_keys:
            t1 = table_dict[t1]
            t2 = table_dict[t2]
        join_dict[t1].add(k1)
        join_dict[t2].add(k2)
    return join_dict


def _try_parse_literal(s):
    try:
        ret = ast.literal_eval(s)
        # IN needs a tuple operand
        # String equality needs a string operand
        if isinstance(ret, tuple) or isinstance(ret, str):
            return ret
        return s
    except:
        return s


def _get_predicate_dict(predicates, table_dict):
    predicates = [predicates[x:x + 3] for x in range(0, len(predicates), 3)]
    predicate_dict = {}
    for p in predicates:
        split_p = p[0].split('.')
        table_name = table_dict[split_p[0]]
        if table_name not in predicate_dict:
            predicate_dict[table_name] = {}
            predicate_dict[table_name]['cols'] = []
            predicate_dict[table_name]['ops'] = []
            predicate_dict[table_name]['vals'] = []
        predicate_dict[table_name]['cols'].append(split_p[1])
        predicate_dict[table_name]['ops'].append(p[1])
        predicate_dict[table_name]['vals'].append(_try_parse_literal(p[2]))
    return predicate_dict


def JobToQuery(csv_file, use_alias_keys=True):
    """Parses custom #-delimited query csv.

    `use_alias_keys` only applies to the 2nd return value.
    If use_alias_keys is true, join_dict will use aliases (t, mi) as keys;
    otherwise it uses real table names (title, movie_index).

    Converts into (tables, join dict, predicate dict, true cardinality).  Only
    works for single equivalency class.
    """
    queries = []
    with open(csv_file) as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            reader = csv.reader(row)  # comma-separated
            table_dict = _get_table_dict(next(reader))
            join_dict = _get_join_dict(next(reader), table_dict, use_alias_keys)
            predicate_dict = _get_predicate_dict(next(reader), table_dict)
            true_cardinality = int(next(reader)[0])
            queries.append((list(table_dict.values()), join_dict,
                            predicate_dict, true_cardinality))

        return queries
def FormattingQuery(csv_file,sep, use_alias_keys=True):
    use_alias_keys=False
    queries = []
    with open(csv_file) as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=sep))
        for row in data_raw:
            reader = csv.reader(row)  # comma-separated
            table_dict = _get_table_dict(next(reader))
            join_dict = _get_join_dict(next(reader), table_dict, use_alias_keys)
            predicate_dict = _get_predicate_dict(next(reader), table_dict)
            true_cardinality = int(next(reader)[0])
            queries.append((list(table_dict.values()), join_dict,
                            predicate_dict, true_cardinality))

        return queries

def UnpackQueries(concat_table, queries):
    """Converts custom query representation to (cols, ops, vals)."""
    is_single = not (hasattr(concat_table,'table_names'))

    converted = []
    true_cards = []
    for q in queries:
        tables, join_dict, predicate_dict, true_cardinality = q
        # All predicates in a query (an AND of these).
        query_cols, query_ops, query_vals = [], [], []

        skip = False
        # A naive impl of "is join graph subset of another join" check.
        for table in tables:
            if  (not is_single) and table not in concat_table.table_names:
                print(f'skipping query due to missing table {table}')
                skip = True
                break
            # Add the indicators.
            if is_single:
                continue
            idx = concat_table.ColumnIndex('__in_{}'.format(table))
            query_cols.append(concat_table.columns[idx])
            query_ops.append('=')
            query_vals.append(1)

        if skip:
            continue

        for table, preds in predicate_dict.items():
            cols = preds['cols']
            ops = preds['ops']
            vals = preds['vals']
            assert len(cols) == len(ops) and len(ops) == len(vals)
            for c, o, v in zip(cols, ops, vals):
                if is_single:
                    column = concat_table.columns[concat_table.name_to_index[c]]
                else :
                    index = concat_table.TableColumnIndex(table, c, allow_none=True)

                    if index is None:
                        print(f'skipping query due to missing column {table}.{c}')
                        skip = True
                        break
                    column = concat_table.columns[index]
                    
                query_cols.append(column)
                query_ops.append(o)
                # Cast v into the correct column dtype.
                cast_fn = column.all_distinct_values.dtype.type
                # If v is a collection, cast its elements.
                if isinstance(v, (list, set, tuple)):
                    qv = type(v)(map(cast_fn, v))
                else:
                    try :
                        qv = cast_fn(v)
                    except:
                        if v == 'None':
                            qv = -1
                        else:
                            qv = v

                query_vals.append(qv)
            if skip:
                break

        if skip:
            continue

        converted.append((query_cols, query_ops, query_vals))
        true_cards.append(true_cardinality)
    return converted, true_cards


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the "true" ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def HumanFormat(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.2f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])




def FormattingQuery_JoinFilter(csv_file,schema_join_list,sep,use_alias_keys=True):

    def switch_clause(clause):
        token1, token2 = clause.split('=')
        return f"{token2}={token2}"
    queries = []
    table_dict_list = list()
    with open(csv_file) as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter=sep))
        for row in data_raw:
            reader = csv.reader(row)  # comma-separated
            table_dict = _get_table_dict(next(reader))
            join_dict = _get_join_dict(next(reader), table_dict, use_alias_keys)
            predicate_dict = _get_predicate_dict(next(reader), table_dict)
            true_cardinality = int(next(reader)[0])
            queries.append((list(table_dict.values()), join_dict,
                            predicate_dict, true_cardinality))
            table_dict_list.append(table_dict)

    # not_support_query = ([],dict(),dict(),-1)
    not_support_query_idx = list()
    if schema_join_list is None:
        schema_joins = None
    else:
        schema_joins = join_to_tuple(schema_join_list)
    with open(csv_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            query_joins = get_query_joins(line,sep)
            fitted, new_joins = fit_joins_to_schema(schema_joins, query_joins)
            if not fitted:
                not_support_query_idx.append(i)
    return queries,not_support_query_idx



def get_table(att_w_table):
    return att_w_table.split('.')[0]

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def merge_groups(equi_groups, equi_joins, s_i, t_i, s, t):
    assert s_i != t_i
    # merge two groups
    if s_i < t_i:
        latter_group = equi_groups.pop(t_i)
        former_group = equi_groups.pop(s_i)

        latter_joins = equi_joins.pop(t_i)
        former_joins = equi_joins.pop(s_i)
    else:
        latter_group = equi_groups.pop(s_i)
        former_group = equi_groups.pop(t_i)

        latter_joins = equi_joins.pop(s_i)
        former_joins = equi_joins.pop(t_i)

    new_group = former_group + latter_group
    new_joins = former_joins + latter_joins + [(s,t)]

    equi_groups.append(new_group)
    equi_joins.append(new_joins)

    return equi_groups, equi_joins


def gen_equi_groups(joins):
    attrs = set()
    for s,t in joins:
        attrs.add(s)
        attrs.add(t)

    # make singletons (initial equivalence groups)
    equi_groups = list([[a] for a in attrs])
    equi_joins  = list([[] for a in attrs])
    initial_num_groups = len(equi_groups)

    for s,t in joins:
        # find equivalence groups that contain s and t
        s_i = find_indices(equi_groups, lambda group: s in group)
        t_i = find_indices(equi_groups, lambda group: t in group)
        assert len(s_i) == len(t_i) == 1
        s_i = s_i[0]
        t_i = t_i[0]
        # already considered by FD
        if s_i == t_i:
            continue

        # merge two groups
        equi_groups, equi_joins = merge_groups(equi_groups, equi_joins, s_i, t_i, s, t)
        assert sum(list(map(lambda g: len(g), equi_groups))) == initial_num_groups

    return equi_groups, equi_joins


def fit_joins_to_schema(schema_joins, query_joins):
#     print(f"schema_joins = {schema_joins}")
#     print(f"query_joins  = {query_joins}")
    if schema_joins is None:
        _, query_equi_joins = gen_equi_groups(query_joins)
        query_equi_joins = [item for sublist in query_equi_joins for item in sublist]
        return True,query_equi_joins

    new_joins = [(s,t) for s,t in query_joins if (s,t) in schema_joins or (t,s) in schema_joins]
    fix_joins = [(s,t) for s,t in query_joins if (s,t) not in schema_joins and (t,s) not in schema_joins]

    if len(fix_joins) > 0:
        schema_equi_groups, schema_equi_joins = gen_equi_groups(schema_joins)
#         print(f"schema_equi_groups = {schema_equi_groups}")
#         print(f"schema_equi_joins  = {schema_equi_joins}")
        query_equi_groups, query_equi_joins = gen_equi_groups(new_joins)

        for s,t in fix_joins:
            # first, check if this join is unseen in schema
            s_i = find_indices(schema_equi_groups, lambda group: s in group)
            t_i = find_indices(schema_equi_groups, lambda group: t in group)
            if not(len(s_i) == len(t_i) == 1) :
                return False, 1
            s_i = s_i[0]
            t_i = t_i[0]
            if s_i != t_i:
#                 print("invalid\n")
                return False, None

            s_i = find_indices(query_equi_groups, lambda group: s in group)
            t_i = find_indices(query_equi_groups, lambda group: t in group)
            # s should be added as a singleton
            if len(s_i) == 0:
                query_equi_groups.append([s])
                query_equi_joins.append([])
            if len(t_i) == 0:
                query_equi_groups.append([t])
                query_equi_joins.append([])
#         print(f"query_equi_groups  = {query_equi_groups}")

        for s,t in fix_joins:
            s_i = find_indices(schema_equi_groups, lambda group: s in group)[0]

            cand = None
            # select a join that can merge two groups
            for cand_s,cand_t in schema_equi_joins[s_i]:
                cand_s_i = find_indices(query_equi_groups, lambda group: cand_s in group)
                cand_t_i = find_indices(query_equi_groups, lambda group: cand_t in group)

                if len(cand_s_i) == 0 or len(cand_t_i) == 0:
                    continue

                assert len(cand_s_i) == len(cand_t_i) == 1
#                 print(f"cand_s = {cand_s}, cand_t = {cand_t}, query_equi_groups = {query_equi_groups}")
                cand_s_i = cand_s_i[0]
                cand_t_i = cand_t_i[0]
                if cand_s_i != cand_t_i:
                    if cand is None:
                        cand = cand_s,cand_t
                    else:
                        # no more than one join can merge two groups
                        assert False
                    query_equi_groups, query_equi_joins = \
                    merge_groups(query_equi_groups, query_equi_joins, cand_s_i, cand_t_i, cand_s, cand_t)

            # cannot be replaced by a join in schema
            if cand is None:
                if len(schema_equi_joins[s_i]) == 0:
#                 print("invalid\n")
                    return False, None
                else:
                    continue

            new_joins.append(cand)

#     print(f"valid, new_joins = {new_joins}\n")
    return True, new_joins

def join_to_tuple(join_list):
    result = list()
    for join in join_list:
        A,B = join.split('=')
        result.append((A,B))
    return result
def join_to_list(join_tuple):
    result = list()
    for A,B in join_tuple:
        result.append(f"{A}={B}")
    return result

def tableAliasDict(table_clause):
    alias_dict = dict()
    tokens = table_clause.split(',')
    for token in tokens:
        table_alias = token.split(' ')
        if len(table_alias) == 1:
            alias_dict[table_alias[0]] = table_alias[0]
        elif len(table_alias) ==2:
            alias_dict[table_alias[1]] = table_alias[0]
        else :
            assert False
    return alias_dict
def get_original_form(join_list,alias_dict):
    result = list()
    for join in join_list:
        if len(join) == 0:
            continue
        A,B = join.split('=')
        t1,c1 = A.split('.')
        t2,c2 = B.split('.')
        T1 = alias_dict[t1]
        T2 = alias_dict[t2]
        result.append(f"{T1}.{c1}={T2}.{c2}")
    return result

def get_query_joins(line,sep):
    table_clause = line.split(sep)[0]
    join_clause = line.split(sep)[1]
    alias_dict = tableAliasDict(table_clause)
    if len(join_clause.split(',')) == 1:
        return list()
    join_list = get_original_form(join_clause.split(','),alias_dict)
    query_joins = join_to_tuple(join_list)
    return query_joins


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')
