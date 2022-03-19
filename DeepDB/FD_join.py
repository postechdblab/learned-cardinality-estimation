
def get_table(att_w_table):
    return att_w_table.split('.')[0]

def find_indices(lst, condition):
    return [i for i, elem in enumerate(lst) if condition(elem)]


def merge_groups(equi_groups, equi_joins, s_i, t_i, s, t):
    #equi_groups = copy.deepcopy(equi_groups)
    #equi_joins  = copy.deepcopy(equi_joins)

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
    if len(joins) == 0:
        return [[]], [[]]

    attrs = set()
    for s,t in joins:
        attrs.add(s)
        attrs.add(t)
    assert len(attrs) > 0

    # make singletons (initial equivalence groups)
    equi_groups = [[a] for a in attrs]
    equi_joins  = [[] for a in attrs]

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

tpcds_pks = {'customer_address' : ['ca_address_sk'],
'customer_demographics' : ['cd_demo_sk'],
'date_dim' : ['d_date_sk'],
'warehouse' : ['w_warehouse_sk'],
'ship_mode' : ['sm_ship_mode_sk'],
'time_dim' : ['t_time_sk'],
'reason' : ['r_reason_sk'],
'income_band' : ['ib_income_band_sk'],
'item' : ['i_item_sk'],
'store' : ['s_store_sk'],
'call_center' : ['cc_call_center_sk'],
'customer' : ['c_customer_sk'],
'web_site' : ['web_site_sk'],
#'store_returns' : ['sr_item_sk', 'sr_ticket_number'],
'household_demographics' : ['hd_demo_sk'],
'web_page' : ['wp_web_page_sk'],
'promotion' : ['p_promo_sk'],
'catalog_page' : ['cp_catalog_page_sk']
#'inventory' : ['inv_date_sk', 'inv_item_sk', 'inv_warehouse_sk'],
#'catalog_returns' : ['cr_item_sk', 'cr_order_number'],
#'web_returns' : ['wr_item_sk', 'wr_order_number'],
#'web_sales' : ['ws_item_sk', 'ws_order_number'],
#'catalog_sales' : ['cs_item_sk', 'cs_order_number'],
#'store_sales' : ['ss_item_sk', 'ss_ticket_number']
}

def is_pk(col):
    [t, c] = col.split('.')
    if t in tpcds_pks:
        return tpcds_pks[t][0] == c
    return '.id' in col

def get_pk(cols):
    pks = [col for col in cols if is_pk(col)]
    assert len(pks) == 1
    return pks[0]

def fit_joins_to_schema(schema_joins, query_joins):
    #print(f"schema_joins = {schema_joins}")
    #print(f"query_joins  = {query_joins}")
    if schema_joins is None:
        _, query_equi_joins = gen_equi_groups(query_joins)
        query_equi_joins = [item for sublist in query_equi_joins for item in sublist]
        return True, query_equi_joins, None

    new_joins = [(s,t) for s,t in query_joins if (s,t) in schema_joins or (t,s) in schema_joins]
    fix_joins = [(s,t) for s,t in query_joins if (s,t) not in schema_joins and (t,s) not in schema_joins]
    #print(f"new_joins = {new_joins}")
    #print(f"fix_joins = {fix_joins}")

    added_tables = []

    if len(fix_joins) > 0:

        schema_equi_groups, schema_equi_joins = gen_equi_groups(schema_joins)
        #print(f"schema_equi_groups = {schema_equi_groups}")
        #print(f"schema_equi_joins  = {schema_equi_joins}")
        query_equi_groups, query_equi_joins = gen_equi_groups(new_joins)
        #print(f"query_equi_groups = {query_equi_groups}")
        #print(f"query_equi_joins = {query_equi_joins}")

        for s,t in fix_joins:
            # first, check if this join is unseen in schema
            s_i = find_indices(schema_equi_groups, lambda group: s in group)
            t_i = find_indices(schema_equi_groups, lambda group: t in group)
            if not(len(s_i) == len(t_i) == 1):
                return False, None, None
            s_i = s_i[0]
            t_i = t_i[0]
            if s_i != t_i:
                print("invalid\n")
                return False, None, None

            s_i = find_indices(query_equi_groups, lambda group: s in group)
            t_i = find_indices(query_equi_groups, lambda group: t in group)
            # s should be added as a singleton
            if len(s_i) == 0:
                query_equi_groups.append([s])
                query_equi_joins.append([])
            if len(t_i) == 0:
                query_equi_groups.append([t])
                query_equi_joins.append([])
        #print(f"query_equi_groups  = {query_equi_groups}")

        for s,t in fix_joins:
            s_i = find_indices(schema_equi_groups, lambda group: s in group)[0]
            t_i = find_indices(schema_equi_groups, lambda group: t in group)[0]
            q_s_i = find_indices(query_equi_groups, lambda group: s in group)[0]
            q_t_i = find_indices(query_equi_groups, lambda group: t in group)[0]
            assert s_i == t_i
            # functional dependency
            if q_s_i == q_t_i:
                continue
            #assert q_s_i != q_t_i,(q_s_i,q_t_i,s_i,t_i)

            cand = None
            # select a join that can merge two groups
            for cand_s,cand_t in schema_equi_joins[s_i]:
                if cand_s != s and cand_t != t and cand_s != t and cand_t != s:
                    #print(f'{cand_s}, {cand_t} not a candiate for {s}, {t}')
                    continue

                cand_s_i = find_indices(query_equi_groups, lambda group: cand_s in group)
                cand_t_i = find_indices(query_equi_groups, lambda group: cand_t in group)

                if len(cand_s_i) == 0 or len(cand_t_i) == 0:
                    #print(f'{cand_s}, {cand_t} not in query')
                    continue

                assert len(cand_s_i) == len(cand_t_i) == 1, f'query_equi_groups = {query_equi_groups}, cand_s_i = {cand_s_i}, cand_t_i = {cand_t_i}'
                #print(f"cand_s = {cand_s}, cand_t = {cand_t}, query_equi_groups = {query_equi_groups}")
                cand_s_i = cand_s_i[0]
                cand_t_i = cand_t_i[0]

                if (cand_s_i == q_s_i and cand_t_i == q_t_i) or (cand_s_i == q_t_i and cand_t_i == q_s_i):
                #if cand_s_i != cand_t_i:
                    #print(f'{cand_s}, {cand_t} is a candidate for {s}, {t}')
                    if cand is None:
                        cand = cand_s,cand_t
                    else:
                        # no more than one join can merge two groups
                        assert False
                    query_equi_groups, query_equi_joins = \
                    merge_groups(query_equi_groups, query_equi_joins, cand_s_i, cand_t_i, cand_s, cand_t)
                    #q_s_i = find_indices(query_equi_groups, lambda group: s in group)[0]
                    #q_t_i = find_indices(query_equi_groups, lambda group: t in group)[0]

            # cannot be replaced by a join in schema
            if cand is None:
                #if len(schema_equi_joins[s_i]) == 0:
                #    print("invalid\n")
                #    return False, None
                #else:
                #    continue

                # if FK-FK, replace it with two PK-FK joins
                if is_pk(s) or is_pk(t):
#                     print("no cand, invalid\n")
                    return False, None, None
                else:
                    #print(f"query_equi_groups = {query_equi_groups}")
                    #print(f"query_equi_joins = {query_equi_joins}")

                    pk = get_pk(schema_equi_groups[s_i])
                    new_joins.append((pk, s))
                    new_joins.append((pk, t))
#                     print(f'no cand, replaced FK-FK with {pk}={s}, {pk}={t}')

                    #XXX pk can be already contained in some group
                    pk_i = find_indices(query_equi_groups, lambda group: pk in group)
                    assert len(pk_i) <= 1

                    if len(pk_i) == 0:
                        query_equi_groups[q_s_i].append(pk)
                        query_equi_joins[q_s_i].append((pk, s))
                    else:
                        query_equi_groups, query_equi_joins = \
                        merge_groups(query_equi_groups, query_equi_joins, q_s_i, pk_i[0], s, pk)

                        q_s_i = find_indices(query_equi_groups, lambda group: s in group)[0]
                        q_t_i = find_indices(query_equi_groups, lambda group: t in group)[0]

                    #print(f'append join at {q_s_i}: {query_equi_joins[q_s_i]}')
                    #print(f"query_equi_groups = {query_equi_groups}")
                    #print(f"query_equi_joins = {query_equi_joins}")

                    query_equi_groups, query_equi_joins = \
                    merge_groups(query_equi_groups, query_equi_joins, q_s_i, q_t_i, s, t)
                    #print(f"query_equi_groups = {query_equi_groups}")
                    #print(f"query_equi_joins = {query_equi_joins}")

                    added_table = pk.split('.')[0]
                    if added_table not in added_tables:
                        added_tables.append(added_table)
                    #print(f'added_tables = {added_tables}')

                    continue

            new_joins.append(cand)
#             print(f'{cand} replaced {s}, {t}')
#     print(f"valid, new_joins = {new_joins}\n")
    return True, new_joins, added_tables

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
def get_original_form(join_list):
    result = list()
    for join in join_list:
        if len(join) == 0:
            continue
        A,B = join.split('=')
        t1,c1 = A.split('.')
        t2,c2 = B.split('.')
        result.append(f"{T1}.{c1} = {T2}.{c2}")
    return result

def join_to_set(join_tuple):
    result = set()
    for A,B in join_tuple:
        result.add(f"{A} = {B}")
    return result

def applyFD(Query,schema):
    query_join_list = Query.fd_join_list
    query_joins = join_to_tuple(query_join_list)

    
    schema_join_list = schema.relationship_dictionary.keys()
    schema_join_list = [s_join.replace(' ','') for s_join in schema_join_list]
    schema_joins = join_to_tuple(schema_join_list)

    fitted, new_joins, new_tables = fit_joins_to_schema(schema_joins, query_joins)
    if not fitted:
        return False
    
    if new_tables is not None:
        for new_table in new_tables:
            Query.table_set.add(new_table)

    Query.relationship_set = set()
    for left_part, right_part in new_joins:
        if left_part + ' = ' + right_part in schema.relationship_dictionary.keys():
            Query.add_join_condition(left_part + ' = ' + right_part)
        elif right_part + ' = ' + left_part in schema.relationship_dictionary.keys():
            Query.add_join_condition(right_part + ' = ' + left_part)
            
    return True