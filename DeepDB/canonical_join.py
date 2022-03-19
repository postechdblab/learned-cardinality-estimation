
# def get_table(att_w_table):
#     return att_w_table.split('.')[0]

# def find_indices(lst, condition):
#     return [i for i, elem in enumerate(lst) if condition(elem)]


# def merge_groups(equi_groups, equi_joins, s_i, t_i, s, t):
#     assert s_i != t_i
#     # merge two groups
#     if s_i < t_i:
#         latter_group = equi_groups.pop(t_i)
#         former_group = equi_groups.pop(s_i)

#         latter_joins = equi_joins.pop(t_i) 
#         former_joins = equi_joins.pop(s_i) 
#     else:
#         latter_group = equi_groups.pop(s_i)
#         former_group = equi_groups.pop(t_i)

#         latter_joins = equi_joins.pop(s_i) 
#         former_joins = equi_joins.pop(t_i) 

#     new_group = former_group + latter_group
#     new_joins = former_joins + latter_joins + [(s,t)]

#     equi_groups.append(new_group)
#     equi_joins.append(new_joins)

#     return equi_groups, equi_joins


# def gen_equi_groups(joins):
#     attrs = set()
#     for s,t in joins:
#         attrs.add(s)
#         attrs.add(t)

#     # make singletons (initial equivalence groups)
#     equi_groups = list([[a] for a in attrs])
#     equi_joins  = list([[] for a in attrs])
#     initial_num_groups = len(equi_groups)

#     for s,t in joins:
#         # find equivalence groups that contain s and t
#         s_i = find_indices(equi_groups, lambda group: s in group)
#         t_i = find_indices(equi_groups, lambda group: t in group)
#         assert len(s_i) == len(t_i) == 1
#         s_i = s_i[0]
#         t_i = t_i[0]
#         # already considered by FD
#         if s_i == t_i:
#             continue

#         # merge two groups
#         equi_groups, equi_joins = merge_groups(equi_groups, equi_joins, s_i, t_i, s, t)
#         assert sum(list(map(lambda g: len(g), equi_groups))) == initial_num_groups 

#     return equi_groups, equi_joins


# def fit_joins_to_schema(schema_joins, query_joins): 
#     print(f"schema_joins = {schema_joins}\n")
#     print(f"query_joins  = {query_joins}\n")

#     # just remove FDs in query_joins
#     if schema_joins is None:
#         _, query_equi_joins = gen_equi_groups(query_joins)
#         query_equi_joins = [item for sublist in query_equi_joins for item in sublist]
#         print(f"query_equi_joins = {query_equi_joins}")
#         return True, query_equi_joins 

#     new_joins = [(s,t) for s,t in query_joins if (s,t) in schema_joins or (t,s) in schema_joins]
#     fix_joins = [(s,t) for s,t in query_joins if (s,t) not in schema_joins and (t,s) not in schema_joins]
#     schema_equi_groups, schema_equi_joins = gen_equi_groups(schema_joins)

#     if len(fix_joins) > 0:
#         print(f"schema_equi_groups = {schema_equi_groups}\n")
#         print(f"schema_equi_joins  = {schema_equi_joins}\n")
#         query_equi_groups, query_equi_joins = gen_equi_groups(new_joins)
        
#         for s,t in fix_joins:
#             # first, check if this join is unseen in schema
#             s_i = find_indices(schema_equi_groups, lambda group: s in group)
#             t_i = find_indices(schema_equi_groups, lambda group: t in group)
#             assert len(s_i) == len(t_i) == 1, f"s_i = {s_i}, t_i = {t_i}, schema_equi_groups = {schema_equi_groups}, s = {s}, t = {t}"
#             s_i = s_i[0]
#             t_i = t_i[0]
#             if s_i != t_i:
#                 print("invalid, not in the schema\n")
#                 return False, None

#             s_i = find_indices(query_equi_groups, lambda group: s in group)
#             t_i = find_indices(query_equi_groups, lambda group: t in group)
#             # s should be added as a singleton 
#             if len(s_i) == 0:
#                 query_equi_groups.append([s])
#                 query_equi_joins.append([])
#             if len(t_i) == 0:
#                 query_equi_groups.append([t])
#                 query_equi_joins.append([])
#         print(f"query_equi_groups  = {query_equi_groups}\n")

#         for s,t in fix_joins:
#             s_i = find_indices(schema_equi_groups, lambda group: s in group)[0]

#             cand = None
#             # select a join that can merge two groups
#             for cand_s,cand_t in schema_equi_joins[s_i]:
#                 cand_s_i = find_indices(query_equi_groups, lambda group: cand_s in group)
#                 cand_t_i = find_indices(query_equi_groups, lambda group: cand_t in group)

#                 if len(cand_s_i) == 0 or len(cand_t_i) == 0:
#                     continue

#                 assert len(cand_s_i) == len(cand_t_i) == 1
#                 print(f"cand_s = {cand_s}, cand_t = {cand_t}, query_equi_groups = {query_equi_groups}\n")
#                 cand_s_i = cand_s_i[0]
#                 cand_t_i = cand_t_i[0]
#                 if cand_s_i != cand_t_i:
#                     if cand is None:
#                         cand = cand_s,cand_t
#                     else:
#                         # no more than one join can merge two groups 
#                         assert False
#                     query_equi_groups, query_equi_joins = \
#                     merge_groups(query_equi_groups, query_equi_joins, cand_s_i, cand_t_i, cand_s, cand_t)

#             # cannot be replaced by a join in schema, or can be ignored by FDs
#             if cand is None:
#                 if len(schema_equi_joins[s_i]) == 0:
#                     print("invalid")
#                     return False, None
#                 else:
#                     print(f"cannot replace ({s}, {t})\n")
#                     continue

#             new_joins.append(cand)
#     else:
#         print("no joins to fix")

#     print(f"valid, new_joins = {new_joins}\n")
#     return True, new_joins

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
            t_i = find_indices(schema_equi_groups, lambda group: t in group)[0]
            q_s_i = find_indices(query_equi_groups, lambda group: s in group)[0]
            q_t_i = find_indices(query_equi_groups, lambda group: t in group)[0]
            assert s_i == t_i
            assert q_s_i != q_t_i

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

                assert len(cand_s_i) == len(cand_t_i) == 1
#                 print(f"cand_s = {cand_s}, cand_t = {cand_t}, query_equi_groups = {query_equi_groups}")
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
                if len(schema_equi_joins[s_i]) == 0:
#                 print("invalid\n")
                    return False, None
                else:
                    continue

            new_joins.append(cand)
            print(f'{cand} replaced {s}, {t}')

#     print(f"valid, new_joins = {new_joins}\n")
    return True, new_joins

# join = pair of table.col

# test case 1
schema_joins = [('A.a','B.b'), ('A.a','C.c')]
query_joins = [('B.b','C.c'), ('A.a', 'B.b')]

#fitted, new_joins = fit_joins_to_schema(schema_joins, query_joins)
#assert fitted == True

# test case 2
query_joins = [('B.b','C.c')]
#fitted, new_joins = fit_joins_to_schema(schema_joins, query_joins)
#assert fitted == False

# test case 3
query_joins = [('A.a','C.c')]
#fitted, new_joins = fit_joins_to_schema(schema_joins, query_joins)
#assert fitted == True 


'''
query_joins = [('title.id', 'movie_companies.movie_id'),
        ('title.id', 'cast_info.movie_id'),
        ('movie_companies.movie_id', 'cast_info.movie_id'),
        ('movie_companies.company_id', 'company_name.id'),
        ('movie_companies.company_type_id', 'company_type.id')]

schema_joins = [('title.id', 'aka_title.movie_id'),
        ('title.id', 'cast_info.movie_id'),
        ('title.id', 'complete_cast.movie_id'),
        ('title.id', 'movie_companies.movie_id'),
        ('title.id', 'movie_info.movie_id'),
        ('title.id', 'movie_info_idx.movie_id'),
        ('title.id', 'movie_keyword.movie_id'),
        ('title.id', 'movie_link.movie_id'),
        ('title.kind_id', 'kind_type.id'),
        ('comp_cast_type.id', 'complete_cast.subject_id'),
        ('company_name.id', 'movie_companies.company_id'),
        ('company_type.id', 'movie_companies.company_type_id'),
        ('movie_info_idx.info_type_id', 'info_type.id'),
        ('keyword.id', 'movie_keyword.keyword_id'),
        ('link_type.id', 'movie_link.link_type_id')] 
fitted, new_joins = fit_joins_to_schema(schema_joins, query_joins)
'''

#schema_joins = [('item.i_item_sk', 'catalog_returns.cr_item_sk'), ('item.i_item_sk', 'catalog_sales.cs_item_sk'), ('item.i_item_sk', 'inventory.inv_item_sk'), ('item.i_item_sk', 'store_returns.sr_item_sk'), ('item.i_item_sk', 'store_sales.ss_item_sk'), ('item.i_item_sk', 'web_returns.wr_item_sk'), ('item.i_item_sk', 'web_sales.ws_item_sk'), ('catalog_returns.cr_call_center_sk', 'call_center.cc_call_center_sk'), ('catalog_returns.cr_catalog_page_sk', 'catalog_page.cp_catalog_page_sk'), ('catalog_returns.cr_refunded_addr_sk', 'customer_address.ca_address_sk'), ('catalog_returns.cr_returned_date_sk', 'date_dim.d_date_sk'), ('catalog_returns.cr_returning_customer_sk', 'customer.c_customer_sk'), ('catalog_sales.cs_bill_cdemo_sk', 'customer_demographics.cd_demo_sk'), ('catalog_sales.cs_bill_hdemo_sk', 'household_demographics.hd_demo_sk'), ('catalog_sales.cs_promo_sk', 'promotion.p_promo_sk'), ('catalog_sales.cs_ship_mode_sk', 'ship_mode.sm_ship_mode_sk'), ('catalog_sales.cs_sold_time_sk', 'time_dim.t_time_sk'), ('catalog_sales.cs_warehouse_sk', 'warehouse.w_warehouse_sk'), ('store_returns.sr_reason_sk', 'reason.r_reason_sk'), ('store_returns.sr_store_sk', 'store.s_store_sk'), ('web_returns.wr_web_page_sk', 'web_page.wp_web_page_sk'), ('web_sales.ws_web_site_sk', 'web_site.web_site_sk'), ('household_demographics.hd_income_band_sk', 'income_band.ib_income_band_sk')]

#query_joins = [('customer.c_customer_sk', 'store_sales.ss_customer_sk'), ('customer_address.ca_address_sk', 'store_sales.ss_addr_sk'), ('date_dim.d_date_sk', 'store_sales.ss_sold_date_sk'), ('item.i_item_sk', 'store_sales.ss_item_sk'), ('promotion.p_promo_sk', 'store_sales.ss_promo_sk')]

#schema_joins = [('A.a', 'B.b'), ('B.b', 'C.c'), ('A.a', 'C.c')]
schema_joins = None
query_joins = [('A.a', 'B.b'), ('B.b', 'C.c'), ('A.a', 'C.c')]
fitted, new_joins = fit_joins_to_schema(schema_joins, query_joins)
