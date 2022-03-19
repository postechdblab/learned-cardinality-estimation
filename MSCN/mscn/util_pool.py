from mscn.util_common import *


def encode_data(tables, samples, predicates, joins, column_min_max_vals, table2vec, column2vec, op2vec, join2vec, string_columns = set(), word_vectors_path = None, is_imdb = False, added_tables = None):
    features_enc = []
    masks = []
    num_tables = len(table2vec)
    if word_vectors_path == None:
        word_vectors = dict()
    else:
        word_vectors = KeyedVectors.load(word_vectors_path, mmap='r')

    table_idx_map = [t.split(" ")[1] for t in table2vec.keys()]
    table_idx_map = {t:i for i,t in enumerate(table_idx_map)}

    table_idx_map = dict()
    column_idx_maps = dict()
    column_sizes = dict()
    join_idx_maps = dict()
    pred_sizes = dict()
    feature_sizes = list()
    
    for col in column2vec:
        if col in string_columns:
            column_sizes[col] = STR_EMB_SIZE
        else:
            column_sizes[col] = 1
    op_size = len(op2vec)
    
    for table_name in table2vec:
        table_alias = table_name.split(" ")[1]
        table_idx_map[table_alias] = len(table_idx_map)
        column_idx_maps[table_alias] = dict()
        pred_size = 0
        for col in column2vec:
            if table_alias == col.split(".")[0]:
                column_idx_maps[table_alias][col] = pred_size
                pred_size += (op_size + column_sizes[col])
        pred_sizes[table_alias] = pred_size
        
        join_idx_maps[table_alias] = dict()
        for join in join2vec:
            if len(join) ==0 or table_alias == join.split("=")[0].split(".")[0] or table_alias == join.split("=")[1].split(".")[0]:
                join_idx_maps[table_alias][join] = len(join_idx_maps[table_alias])
        feature_size = NUM_MATERIALIZED_SAMPLES + pred_size + len(join_idx_maps[table_alias])
        feature_sizes.append(feature_size)
    
    for i, query in enumerate(tables):
        samples_enc = list()
        predicates_enc = list()
        joins_enc = list()


        if added_tables != None:
            new_tables = set(added_tables[i]) - set(query)
        else:
            new_tables = set()

        for table in table_idx_map:
            # initialize per table feature
            samples_enc.append(np.zeros(NUM_MATERIALIZED_SAMPLES, dtype=np.float32))
            predicates_enc.append(np.zeros(pred_sizes[table], dtype=np.float32))
            joins_enc.append(np.zeros(len(join_idx_maps[table]), dtype=np.float32))


        masks.append(np.zeros(num_tables, dtype=np.float32))
        
        for j, table in enumerate(query):
            table_idx = table_idx_map[table.split(" ")[1]]
            samples_enc[table_idx] = samples[i][j]
            masks[i][table_idx] = 1

        for new_table in new_tables:
            table_idx = table_idx_map[new_table.split(" ")[1]]
            samples_enc[table_idx] = np.ones(NUM_MATERIALIZED_SAMPLES, dtype=np.float32)
            masks[i][table_idx] = 1

        for predicate in predicates[i]:
            if len(predicate) == 3:
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]

                pred_vec = []
                pred_vec.append(op2vec[operator])
                if column in string_columns:
                    operand_vec = np.zeros(STR_EMB_SIZE, dtype=np.float32)
                    if operator in {'IN', 'NOT_IN'}:
                        assert (len(val) > 2)
                        val = val[1:-1] #remove '(' and ')'
                        vals = re.split(r",(?=')", val) #split on commas but ignore ones in single quotes
                        
                        for val in vals:
                            new_vec = get_string_embedding(word_vectors, column, val[1:-1], is_imdb)
                            operand_vec = operand_vec + new_vec
                        cnt = len(vals)
                        operand_vec = operand_vec / cnt

                    elif operator in {'LIKE', 'NOT_LIKE'}:
                        cnt = 0
                        for v in val.split('%'):
                            if(len(v) > 0):
                                new_vec = get_string_embedding(word_vectors, column, v, is_imdb)
                                operand_vec = operand_vec + new_vec
                                cnt += 1
                        operand_vec = operand_vec / cnt
                    else:
                        operand_vec = get_string_embedding(word_vectors, column, val, is_imdb)
                    pred_vec.append(operand_vec)
                elif column in DATE_COLUMNS:
                    norm_val = normalize_date(val, 1)
                    pred_vec.append(norm_val)
                else:
                    norm_val = normalize_data(val, column, column_min_max_vals, 1)
                    pred_vec.append(norm_val)
                pred_vec = np.hstack(pred_vec)

                table = column.split(".")[0]
                table_idx = table_idx_map[table]
                column_idx = column_idx_maps[table][column]
                # assume 1 predicate per 1 column at most
                predicates_enc[table_idx][column_idx : column_idx + len(pred_vec)] += pred_vec
        for join in joins[i]:
            # Join instruction
            if len(join) == 0:
                table = query[0].split(" ")[1]
                table_idx = table_idx_map[table]
                join_idx = join_idx_maps[table][join]
                joins_enc[table_idx][join_idx] = 1
            else:
                table1, table2 = join.split("=")[0].split(".")[0], join.split("=")[1].split(".")[0]
                table1_idx, table2_idx = table_idx_map[table1], table_idx_map[table2]
                join1_idx, join2_idx = join_idx_maps[table1][join], join_idx_maps[table2][join]
                joins_enc[table1_idx][join1_idx] = 1
                joins_enc[table2_idx][join2_idx] = 1
        feature_enc = list()
        for i in range(0, len(samples_enc)):
            feature_enc.append(np.hstack([samples_enc[i],predicates_enc[i], joins_enc[i]]))
        feature_enc = np.hstack(feature_enc)
        features_enc.append(feature_enc)
    return features_enc, masks, feature_sizes

