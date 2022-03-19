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

tpcds_pks = {'ca' : ['ca_address_sk'],
'cd' : ['cd_demo_sk'],
'd' : ['d_date_sk'],
'w' : ['w_warehouse_sk'],
'sm' : ['sm_ship_mode_sk'],
't' : ['t_time_sk'],
'r' : ['r_reason_sk'],
'ib' : ['ib_income_band_sk'],
'i' : ['i_item_sk'],
's' : ['s_store_sk'],
'cc' : ['cc_call_center_sk'],
'c' : ['c_customer_sk'],
'web' : ['web_site_sk'],
#'store_returns' : ['sr_item_sk', 'sr_ticket_number'],
'hd' : ['hd_demo_sk'],
'wp' : ['wp_web_page_sk'],
'p' : ['p_promo_sk'],
'cp' : ['cp_catalog_page_sk']
#'inventory' : ['inv_date_sk', 'inv_item_sk', 'inv_warehouse_sk'],
#'catalog_returns' : ['cr_item_sk', 'cr_order_number'],
#'web_returns' : ['wr_item_sk', 'wr_order_number'],
#'web_sales' : ['ws_item_sk', 'ws_order_number'],
#'catalog_sales' : ['cs_item_sk', 'cs_order_number'],
#'store_sales' : ['ss_item_sk', 'ss_ticket_number']
}

def is_pk(col):
    [t, c] = col.split('.')
    if t not in tpcds_pks or t == "t" or t == "cc":
        return '.id' in col
    else:
        return tpcds_pks[t][0] == c

def get_pk(cols):
    pks = [col for col in cols if is_pk(col)]
    assert len(pks) == 1
    return pks[0]

def fit_joins_to_schema(schema_joins, query_joins): 
    if schema_joins is None:
        _, query_equi_joins = gen_equi_groups(query_joins)
        query_equi_joins = [item for sublist in query_equi_joins for item in sublist]
        return True, query_equi_joins, None
    
    new_joins = []
    fix_joins = []
    for (s,t) in query_joins:
        if (s,t) in schema_joins:
            new_joins.append((s,t))
        elif (t,s) in schema_joins:
            new_joins.append((t,s))
        else:
            fix_joins.append((s,t))

    added_tables = []

    if len(fix_joins) > 0:
        schema_equi_groups, schema_equi_joins = gen_equi_groups(schema_joins)
        query_equi_groups, query_equi_joins = gen_equi_groups(new_joins)
        
        for s,t in fix_joins:
            # first, check if this join is unseen in schema
            s_i = find_indices(schema_equi_groups, lambda group: s in group)
            t_i = find_indices(schema_equi_groups, lambda group: t in group)
            # assert len(s_i) == len(t_i) == 1
            if not(len(s_i) == len(t_i) == 1):
                return False, None, None
                
            s_i = s_i[0]
            t_i = t_i[0]
            if s_i != t_i:
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
                    continue

                cand_s_i = find_indices(query_equi_groups, lambda group: cand_s in group)
                cand_t_i = find_indices(query_equi_groups, lambda group: cand_t in group)

                if len(cand_s_i) == 0 or len(cand_t_i) == 0:
                    continue

                assert len(cand_s_i) == len(cand_t_i) == 1, f'query_equi_groups = {query_equi_groups}, cand_s_i = {cand_s_i}, cand_t_i = {cand_t_i}'
                cand_s_i = cand_s_i[0]
                cand_t_i = cand_t_i[0]

                if (cand_s_i == q_s_i and cand_t_i == q_t_i) or (cand_s_i == q_t_i and cand_t_i == q_s_i):
                    if cand is None:
                        cand = cand_s,cand_t
                    else:
                        # no more than one join can merge two groups
                        assert False
                    query_equi_groups, query_equi_joins = \
                    merge_groups(query_equi_groups, query_equi_joins, cand_s_i, cand_t_i, cand_s, cand_t)

            # cannot be replaced by a join in schema
            if cand is None:
                # if FK-FK, replace it with two PK-FK joins
                if is_pk(s) or is_pk(t):
                    print("no cand, invalid\n")
                    return False, None, None
                else:
                    pk = get_pk(schema_equi_groups[s_i])
                    if (pk,s) in schema_equi_joins[s_i]:
                        new_joins.append((pk, s))
                    elif (s,pk) in schema_equi_joins[s_i]:
                        new_joins.append((s, pk))
                    else:
                        return False, None, None
                    if (pk,t) in schema_equi_joins[s_i]:
                        new_joins.append((pk, t))
                    elif (t,pk) in schema_equi_joins[s_i]:
                        new_joins.append((t, pk))
                    else:
                        return False, None, None
                    
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

                    query_equi_groups, query_equi_joins = \
                    merge_groups(query_equi_groups, query_equi_joins, q_s_i, q_t_i, s, t)

                    added_table = pk.split('.')[0]
                    if added_table not in added_tables:
                        added_tables.append(added_table)

                    continue

            new_joins.append(cand)
    return True, new_joins, added_tables 


def join_to_tuple(join_list):
    result = list()
    for join in join_list:
        if len(join) == 0 : continue
        A,B = join.split('=')
        result.append((A,B))
    return result


def tuple_to_join(tuple_list):
    result = list()
    for (A,B) in tuple_list:
        join = "=".join([A,B])
        result.append(join)
    return result

