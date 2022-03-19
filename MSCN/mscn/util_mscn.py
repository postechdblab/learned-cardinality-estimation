from mscn.util_common import * 


def encode_samples(tables, samples, table2vec, added_tables = None):
    samples_enc = []
    for i, query in enumerate(tables):
        samples_enc.append(list())
        if added_tables == None:
            new_tables = set()
        else:
            new_tables = set(added_tables[i]) - set(query)
        for j, table in enumerate(query):
            sample_vec = []
            # Append table one-hot vector
            sample_vec.append(table2vec[table])
            # Append bit vector-
            sample_vec.append(samples[i][j])
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)
        for new_table in new_tables:
            sample_vec = []
            sample_vec.append(table2vec[new_table])
            sample_vec.append(np.ones(NUM_MATERIALIZED_SAMPLES, dtype=np.float32))
            sample_vec = np.hstack(sample_vec)
            samples_enc[i].append(sample_vec)
    return samples_enc


def encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec):
    predicates_enc = []
    joins_enc = []
    for i, query in enumerate(predicates):
        predicates_enc.append(list())
        joins_enc.append(list())
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                norm_val = normalize_data(val, column, column_min_max_vals)

                pred_vec = []
                pred_vec.append(column2vec[column])
                pred_vec.append(op2vec[operator])
                pred_vec.append(norm_val)
                pred_vec = np.hstack(pred_vec)
            else:
                pred_vec = np.zeros((len(column2vec) + len(op2vec) + 1), dtype=np.float32)

            predicates_enc[i].append(pred_vec)
        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i].append(join_vec)
    return predicates_enc, joins_enc


def encode_data_with_string(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec, string_columns, word_vectors_path, is_imdb):
    predicates_enc = []
    joins_enc = []

    word_vectors = KeyedVectors.load(word_vectors_path, mmap='r')

    pred_enc_size = len(column2vec) + len(op2vec) + STR_EMB_SIZE

    for i, query in enumerate(predicates):
        predicates_enc.append(list())
        joins_enc.append(list())
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]

                pred_vec = []
                pred_vec.append(column2vec[column])
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
                    norm_val = normalize_date(val, STR_EMB_SIZE)
                    pred_vec.append(norm_val)
                else:
                    norm_val = normalize_data(val, column, column_min_max_vals, STR_EMB_SIZE)
                    assert len(norm_val) == STR_EMB_SIZE
                    pred_vec.append(norm_val)
                pred_vec = np.hstack(pred_vec)
                
                if len(pred_vec) != pred_enc_size:
                    raise

            else:
                pred_vec = np.zeros((len(column2vec) + len(op2vec) + STR_EMB_SIZE), dtype=np.float32)

            predicates_enc[i].append(pred_vec)
        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i].append(join_vec)
    return predicates_enc, joins_enc
    

