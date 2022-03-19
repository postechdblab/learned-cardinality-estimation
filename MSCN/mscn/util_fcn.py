from mscn.util_common import *


def encode_samples(tables, samples, table2vec, added_tables = None):
    tables_enc = []
    samples_enc = [] 
    num_tables = len(table2vec)
    sample_size = NUM_MATERIALIZED_SAMPLES
    for i, query in enumerate(tables):
        tables_enc.append(np.zeros(num_tables, dtype=np.float32))
        samples_enc.append(np.zeros(num_tables * sample_size, dtype=np.float32))
        if added_tables == None:
            new_tables = set()
        else:
            new_tables = set(added_tables[i]) - set(query)
        for j, table in enumerate(query):
            table_vec = []
            sample_vec = []
            # Append table one-hot vector
            table_vec = table2vec[table]
            table_idx = np.where(table_vec ==1)[0][0]
            
            
            left_padding_size = table_idx
            left_padding = np.zeros(left_padding_size * sample_size, dtype=np.float32)

            # right_padding = np.array([])
            right_padding_size = num_tables - table_idx - 1
            # right_padding = np.pad(right_padding, ((0, right_padding_size),(0, 0)), 'constant')
            right_padding = np.zeros(right_padding_size * sample_size, dtype=np.float32)

            sample_vec = np.hstack([left_padding, samples[i][j], right_padding])

            
            tables_enc[i] = tables_enc[i] + table_vec
            samples_enc[i] = samples_enc[i] + sample_vec
            # samples_enc[i].append(sample_vec)
        for new_table in new_tables:
            table_vec = []
            sample_vec = []
            # Append table one-hot vector
            table_vec = table2vec[table]
            table_idx = np.where(table_vec ==1)[0][0]
            
            left_padding_size = table_idx
            left_padding = np.zeros(left_padding_size * sample_size, dtype=np.float32)

            right_padding_size = num_tables - table_idx - 1
            right_padding = np.zeros(right_padding_size * sample_size, dtype=np.float32)

            sample_vec = np.hstack([left_padding, np.ones(sample_size, dtype=np.float32), right_padding])

            tables_enc[i] = tables_enc[i] + table_vec
            samples_enc[i] = samples_enc[i] + sample_vec
    return tables_enc, samples_enc


def encode_data(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec):
    predicates_enc = []
    joins_enc = []
    num_total_columns = len(column2vec)
    predicate_size = len(op2vec) + 1
    join_size = len(join2vec)
    for i, query in enumerate(predicates):
        predicates_enc.append(np.zeros(num_total_columns * predicate_size, dtype=np.float32))
        joins_enc.append(np.zeros(join_size, dtype=np.float32))
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                norm_val = normalize_data(val, column, column_min_max_vals)

                column_vec = column2vec[column]
                column_idx = np.where(column_vec==1)[0][0]
                left_padding_size = column_idx
                left_padding = np.zeros(predicate_size * left_padding_size, dtype=np.float32)
                right_padding_size = num_total_columns - column_idx - 1 
                right_padding = np.zeros(predicate_size * right_padding_size, dtype=np.float32)
                
                pred_vec = []
                pred_vec.append(op2vec[operator])
                pred_vec.append(norm_val)
                pred_vec = np.hstack(pred_vec)

                pred_vec = np.hstack([left_padding, pred_vec, right_padding])
            else:
                pred_vec = np.zeros(num_total_columns * predicate_size, dtype=np.float32)

            # currently we assume at most 1 predicate for 1 column
            predicates_enc[i] = predicates_enc[i] + pred_vec

        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i] = joins_enc[i] = join_vec
    return predicates_enc, joins_enc


def encode_data_with_string(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec, string_columns, word_vectors_path, is_imdb):
    predicates_enc = []
    joins_enc = []
    num_total_columns = len(column2vec)
    predicate_size = len(op2vec) + STR_EMB_SIZE
    join_size = len(join2vec)
    
    word_vectors = KeyedVectors.load(word_vectors_path, mmap='r')
    
    for i, query in enumerate(predicates):
        predicates_enc.append(np.zeros(num_total_columns * predicate_size, dtype=np.float32))
        joins_enc.append(np.zeros(join_size, dtype=np.float32))
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                
                
                column_vec = column2vec[column]
                column_idx = np.where(column_vec==1)[0][0]
                left_padding_size = column_idx
                left_padding = np.zeros(predicate_size * left_padding_size, dtype=np.float32)
                right_padding_size = num_total_columns - column_idx - 1 
                right_padding = np.zeros(predicate_size * right_padding_size, dtype=np.float32)
                
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
                    norm_val = normalize_date(val, STR_EMB_SIZE)
                    pred_vec.append(norm_val)
                else:
                    norm_val = normalize_data(val, column, column_min_max_vals, STR_EMB_SIZE)
                    pred_vec.append(norm_val)

                pred_vec = np.hstack(pred_vec)

                pred_vec = np.hstack([left_padding, pred_vec, right_padding])
            else:
                pred_vec = np.zeros(num_total_columns * predicate_size, dtype=np.float32)

            # currently we assume at most 1 predicate for 1 column
            predicates_enc[i] = predicates_enc[i] + pred_vec

        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i] = joins_enc[i] = join_vec
    return predicates_enc, joins_enc


def encode_data_with_string_var(predicates, joins, column_min_max_vals, column2vec, op2vec, join2vec, string_columns, word_vectors_path, is_imdb):
    predicates_enc = []
    joins_enc = []
    num_total_columns = len(column2vec)
    join_size = len(join2vec)

    left_padding_sizes = dict()
    total_enc_size = 0
    for idx, column in enumerate(column2vec):
        left_padding_sizes[idx] = total_enc_size
        if column in string_columns:
            operand_size = STR_EMB_SIZE
        else:
            operand_size = 1
        pred_size = len(op2vec) + operand_size
        total_enc_size += pred_size

    word_vectors = KeyedVectors.load(word_vectors_path, mmap='r')
    
    for i, query in enumerate(predicates):
        predicates_enc.append(np.zeros(total_enc_size, dtype=np.float32))
        joins_enc.append(np.zeros(join_size, dtype=np.float32))
        for predicate in query:
            if len(predicate) == 3:
                # Proper predicate
                column = predicate[0]
                operator = predicate[1]
                val = predicate[2]
                
                column_vec = column2vec[column]
                column_idx = np.where(column_vec==1)[0][0]
                left_padding_size = left_padding_sizes[column_idx]
                left_padding = np.zeros(left_padding_size, dtype=np.float32)
                
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
                    norm_val = normalize_data(val, column, column_min_max_vals)
                    pred_vec.append(norm_val)

                pred_vec = np.hstack(pred_vec)

                right_padding_size = total_enc_size - left_padding_size - len(pred_vec)
                assert right_padding_size >= 0, f'right padding size {right_padding_size}, total_enc_size {total_enc_size}, left_padding_size {left_padding_size} pred_vec_size {len(pred_vec)}'
                right_padding = np.zeros(right_padding_size, dtype=np.float32)
                

                pred_vec = np.hstack([left_padding, pred_vec, right_padding])
            else:
                pred_vec = np.zeros(total_enc_size, dtype=np.float32)

            # currently we assume at most 1 predicate for 1 column
            predicates_enc[i] = predicates_enc[i] + pred_vec

        for predicate in joins[i]:
            # Join instruction
            join_vec = join2vec[predicate]
            joins_enc[i] = joins_enc[i] = join_vec
    return predicates_enc, joins_enc