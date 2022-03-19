from ensemble_compilation.graph_representation import SchemaGraph, Table


def gen_syn_single(csv_path):

    schema = SchemaGraph()

    schema.add_table(Table('table0', attributes=['col0', 'col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('table0'),
                           table_size=1000000))
    
    return schema

def gen_syn_multi(csv_path):

    schema = SchemaGraph()

    

    schema.add_table(Table('table0', attributes=['PK'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('table0'),
                           primary_key=['PK'],
                           table_size=1000000))
    

    schema.add_table(Table('table1', attributes=['PK', 'FK'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('table1'),
                           primary_key=['PK'],
                           table_size=1000000))
    

    schema.add_table(Table('table2', attributes=['PK', 'FK'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('table2'),
                           primary_key=['PK'],
                           table_size=1000000))
    

    schema.add_table(Table('table3', attributes=['PK', 'FK'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('table3'),
                           primary_key=['PK'],
                           table_size=1000000))
    

    schema.add_table(Table('table4', attributes=['PK', 'FK'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('table4'),
                           primary_key=['PK'],
                           table_size=1000000))
    

    schema.add_table(Table('table5', attributes=['PK', 'FK'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('table5'),
                           primary_key=['PK'],
                           table_size=1000000))
    

    schema.add_table(Table('table6', attributes=['PK', 'FK'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('table6'),
                           primary_key=['PK'],
                           table_size=1000000))
    

    schema.add_table(Table('table7', attributes=['PK', 'FK'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('table7'),
                           primary_key=['PK'],
                           table_size=1000000))
    

    schema.add_table(Table('table8', attributes=['PK', 'FK'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('table8'),
                           primary_key=['PK'],
                           table_size=1000000))
    

    schema.add_table(Table('table9', attributes=['PK', 'FK'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('table9'),
                           primary_key=['PK'],
                           table_size=1000000))
    

    schema.add_relationship('table1', 'FK', 'table0', 'PK')

    schema.add_relationship('table2', 'FK', 'table1', 'PK')

    schema.add_relationship('table3', 'FK', 'table2', 'PK')

    schema.add_relationship('table4', 'FK', 'table3', 'PK')

    schema.add_relationship('table5', 'FK', 'table4', 'PK')

    schema.add_relationship('table6', 'FK', 'table5', 'PK')

    schema.add_relationship('table7', 'FK', 'table6', 'PK')

    schema.add_relationship('table8', 'FK', 'table7', 'PK')

    schema.add_relationship('table9', 'FK', 'table8', 'PK')
    return schema

def get_syn_multi_table_subset(csv_path,num):
    if num == 2:
        return gen_syn_multi_table2(csv_path)
    elif num == 3:
        return gen_syn_multi_table3(csv_path)
    elif num == 4:
        return gen_syn_multi_table4(csv_path)
    elif num == 5:
        return gen_syn_multi_table5(csv_path)
    elif num == 6:
        return gen_syn_multi_table6(csv_path)
    elif num == 7:
        return gen_syn_multi_table7(csv_path)
    elif num == 8:
        return gen_syn_multi_table8(csv_path)
    elif num == 9:
        return gen_syn_multi_table9(csv_path)
    else:
        assert False


def gen_syn_multi_table2(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('table0', attributes=['PK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table0'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table1', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table1'),primary_key=['PK'],table_size=1000000))
    schema.add_relationship('table1', 'FK', 'table0', 'PK')
    return schema

def gen_syn_multi_table3(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('table0', attributes=['PK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table0'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table1', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table1'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table2', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table2'),primary_key=['PK'],table_size=1000000))           
    
    schema.add_relationship('table1', 'FK', 'table0', 'PK')
    schema.add_relationship('table2', 'FK', 'table1', 'PK')
    return schema

def gen_syn_multi_table4(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('table0', attributes=['PK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table0'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table1', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table1'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table2', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table2'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table3', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table3'),primary_key=['PK'],table_size=1000000))
    
    schema.add_relationship('table1', 'FK', 'table0', 'PK')
    schema.add_relationship('table2', 'FK', 'table1', 'PK')
    schema.add_relationship('table3', 'FK', 'table2', 'PK')
    return schema



def gen_syn_multi_table5(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('table0', attributes=['PK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table0'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table1', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table1'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table2', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table2'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table3', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table3'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table4', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table4'),primary_key=['PK'],table_size=1000000))
   
    schema.add_relationship('table1', 'FK', 'table0', 'PK')
    schema.add_relationship('table2', 'FK', 'table1', 'PK')
    schema.add_relationship('table3', 'FK', 'table2', 'PK')
    schema.add_relationship('table4', 'FK', 'table3', 'PK')
    return schema


def gen_syn_multi_table6(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('table0', attributes=['PK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table0'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table1', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table1'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table2', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table2'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table3', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table3'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table4', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table4'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table5', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table5'),primary_key=['PK'],table_size=1000000))

    schema.add_relationship('table1', 'FK', 'table0', 'PK')
    schema.add_relationship('table2', 'FK', 'table1', 'PK')
    schema.add_relationship('table3', 'FK', 'table2', 'PK')
    schema.add_relationship('table4', 'FK', 'table3', 'PK')
    schema.add_relationship('table5', 'FK', 'table4', 'PK')
    return schema

def gen_syn_multi_table7(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('table0', attributes=['PK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table0'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table1', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table1'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table2', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table2'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table3', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table3'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table4', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table4'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table5', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table5'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table6', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table6'),primary_key=['PK'],table_size=1000000))
    schema.add_relationship('table1', 'FK', 'table0', 'PK')
    schema.add_relationship('table2', 'FK', 'table1', 'PK')
    schema.add_relationship('table3', 'FK', 'table2', 'PK')
    schema.add_relationship('table4', 'FK', 'table3', 'PK')
    schema.add_relationship('table5', 'FK', 'table4', 'PK')
    schema.add_relationship('table6', 'FK', 'table5', 'PK')
    return schema

def gen_syn_multi_table8(csv_path):
    schema = SchemaGraph()

    schema.add_table(Table('table0', attributes=['PK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table0'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table1', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table1'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table2', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table2'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table3', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table3'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table4', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table4'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table5', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table5'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table6', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table6'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table7', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table7'),primary_key=['PK'],table_size=1000000))
    
    schema.add_relationship('table1', 'FK', 'table0', 'PK')
    schema.add_relationship('table2', 'FK', 'table1', 'PK')
    schema.add_relationship('table3', 'FK', 'table2', 'PK')
    schema.add_relationship('table4', 'FK', 'table3', 'PK')
    schema.add_relationship('table5', 'FK', 'table4', 'PK')
    schema.add_relationship('table6', 'FK', 'table5', 'PK')
    schema.add_relationship('table7', 'FK', 'table6', 'PK')

    return schema

def gen_syn_multi_table9(csv_path):
    schema = SchemaGraph()
    schema.add_table(Table('table0', attributes=['PK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table0'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table1', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table1'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table2', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table2'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table3', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table3'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table4', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table4'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table5', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table5'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table6', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table6'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table7', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table7'),primary_key=['PK'],table_size=1000000))
    schema.add_table(Table('table8', attributes=['PK', 'FK'],irrelevant_attributes=[],csv_file_location=csv_path.format('table8'),primary_key=['PK'],table_size=1000000))
    
    schema.add_relationship('table1', 'FK', 'table0', 'PK')
    schema.add_relationship('table2', 'FK', 'table1', 'PK')
    schema.add_relationship('table3', 'FK', 'table2', 'PK')
    schema.add_relationship('table4', 'FK', 'table3', 'PK')
    schema.add_relationship('table5', 'FK', 'table4', 'PK')
    schema.add_relationship('table6', 'FK', 'table5', 'PK')
    schema.add_relationship('table7', 'FK', 'table6', 'PK')
    schema.add_relationship('table8', 'FK', 'table7', 'PK')
    return schema
