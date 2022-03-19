from ensemble_compilation.graph_representation import SchemaGraph, Table

def gen_job_light_imdb_schema(csv_path):
    """
    Just like the full IMDB schema but without tables that are not used in the job-light benchmark.
    """

    schema = SchemaGraph()

    # tables

    # title
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                                                'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                                                'series_years', 'md5sum'],
                           irrelevant_attributes=['episode_of_id', 'title', 'imdb_index', 'phonetic_code', 'season_nr',
                                                  'imdb_id', 'episode_nr', 'series_years', 'md5sum'],
                           no_compression=['kind_id'],
                           csv_file_location=csv_path.format('title'),
                           table_size=3486660))

    # movie_info_idx
    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           irrelevant_attributes=['info', 'note'],
                           no_compression=['info_type_id'],
                           table_size=3147110))

    # movie_info
    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info'),
                           irrelevant_attributes=['info', 'note'],
                           no_compression=['info_type_id'],
                           table_size=24988000))

    # cast_info
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
                                                    'role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           irrelevant_attributes=['nr_order', 'note', 'person_id', 'person_role_id'],
                           no_compression=['role_id'],
                           table_size=63475800))

    # movie_keyword
    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           csv_file_location=csv_path.format('movie_keyword'),
                           no_compression=['keyword_id'],
                           table_size=7522600))

    # movie_companies
    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           csv_file_location=csv_path.format('movie_companies'),
                           irrelevant_attributes=['note'],
                           no_compression=['company_id', 'company_type_id'],
                           table_size=4958300))

    # relationships
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    return schema

def gen_imdb_schema(csv_path):
    """
    Specifies full imdb schema. Also tables not in the job-light benchmark.
    """
    schema = SchemaGraph()

    # tables

    # title
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                                                'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                                                'series_years', 'md5sum'],
                           #irrelevant_attributes=['episode_of_id'],
                           csv_file_location=csv_path.format('title'),
                           table_size=3486660))

    # movie_info_idx
    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           table_size=3147110))

    # movie_info
    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info'),
                           table_size=24988000))

    # info_type
    schema.add_table(Table('info_type', attributes=['id', 'info'],
                           csv_file_location=csv_path.format('info_type'),
                           table_size=113))

    # cast_info
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
                                                    'role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           table_size=63475800))

    # char_name
    schema.add_table(Table('char_name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf',
                                                    'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('char_name'),
                           table_size=4314870))

    # role_type
    schema.add_table(Table('role_type', attributes=['id', 'role'],
                           csv_file_location=csv_path.format('role_type'),
                           table_size=0))

    # complete_cast
    schema.add_table(Table('complete_cast', attributes=['id', 'movie_id', 'subject_id', 'status_id'],
                           csv_file_location=csv_path.format('complete_cast'),
                           table_size=135086))

    # comp_cast_type
    schema.add_table(Table('comp_cast_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('comp_cast_type'),
                           table_size=0))

    # name
    schema.add_table(Table('name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf',
                                               'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('name'),
                           table_size=6379740))

    # aka_name
    schema.add_table(Table('aka_name', attributes=['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf',
                                                   'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           csv_file_location=csv_path.format('aka_name'),
                           table_size=1312270))

    # movie_link, is empty
    schema.add_table(Table('movie_link', attributes=['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
                            csv_file_location=csv_path.format('movie_link')))

    # link_type, no relationships
    schema.add_table(Table('link_type', attributes=['id', 'link'],
                            csv_file_location=csv_path.format('link_type')))

    # movie_keyword
    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           csv_file_location=csv_path.format('movie_keyword'),
                           table_size=7522600))

    # keyword
    schema.add_table(Table('keyword', attributes=['id', 'keyword', 'phonetic_code'],
                           csv_file_location=csv_path.format('keyword'),
                           table_size=236627))

    # person_info
    schema.add_table(Table('person_info', attributes=['id', 'person_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('person_info'),
                           table_size=4130210))

    # movie_companies
    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           csv_file_location=csv_path.format('movie_companies'),
                           table_size=4958300))

    # company_name
    schema.add_table(Table('company_name', attributes=['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf',
                                                       'name_pcode_sf', 'md5sum'],
                           csv_file_location=csv_path.format('company_name'),
                           table_size=362131))

    # company_type
    schema.add_table(Table('company_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('company_type'),
                           table_size=0))

    # aka_title
    schema.add_table(Table('aka_title', attributes=['id', 'movie_id', 'title', 'imdb_index', 'kind_id',
                                                    'production_year', 'phonetic_code', 'episode_of_id', 'season_nr',
                                                    'episode_nr', 'note', 'md5sum'],
                           #irrelevant_attributes=['episode_of_id'],
                           csv_file_location=csv_path.format('aka_title'),
                           table_size=528268))

    # kind_type
    schema.add_table(Table('kind_type', attributes=['id', 'kind'],
                           csv_file_location=csv_path.format('kind_type'),
                           table_size=0))

    #XXX 21 tables
    # relationships

    # title
    # omit self-join for now
    # schema.add_relationship('title', 'episode_of_id', 'title', 'id')
    schema.add_relationship('title', 'kind_id', 'kind_type', 'id') #9

    # movie_info_idx
    #schema.add_relationship('movie_info_idx', 'info_type_id', 'info_type', 'id')
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id') #6

    # movie_info
    schema.add_relationship('movie_info', 'info_type_id', 'info_type', 'id') #17
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id') #5

    # info_type, no relationships

    # cast_info
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id') #2
    schema.add_relationship('cast_info', 'person_id', 'name', 'id') #13
    schema.add_relationship('cast_info', 'person_id', 'aka_name', 'id') #12
    schema.add_relationship('cast_info', 'person_role_id', 'char_name', 'id') #10
    schema.add_relationship('cast_info', 'role_id', 'role_type', 'id') #11

    # char_name, no relationships

    # role_type, no relationships

    # complete_cast
    schema.add_relationship('complete_cast', 'movie_id', 'title', 'id') #3
    #schema.add_relationship('complete_cast', 'status_id', 'comp_cast_type', 'id')
    schema.add_relationship('complete_cast', 'subject_id', 'comp_cast_type', 'id') #14

    # comp_cast_type, no relationships

    # name, no relationships

    # aka_name
    #schema.add_relationship('aka_name', 'person_id', 'name', 'id')

    # movie_link, is empty
    schema.add_relationship('movie_link', 'link_type_id', 'link_type', 'id') #19
    #schema.add_relationship('movie_link', 'linked_movie_id', 'title', 'id')
    #schema.add_relationship('movie_link', 'movie_id', 'title', 'id') #XXX movie_link.id??
    schema.add_relationship('movie_link', 'id', 'title', 'id') #8

    # link_type, no relationships

    # movie_keyword
    schema.add_relationship('movie_keyword', 'keyword_id', 'keyword', 'id') #18
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id') #7

    # keyword, no relationships

    # person_info
    schema.add_relationship('person_info', 'info_type_id', 'info_type', 'id') #20
    #schema.add_relationship('person_info', 'person_id', 'name', 'id')

    # movie_companies
    schema.add_relationship('movie_companies', 'company_id', 'company_name', 'id') #15
    schema.add_relationship('movie_companies', 'company_type_id', 'company_type', 'id') #16
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id') #4

    # company_name, no relationships

    # company_type, no relationships

    # aka_title
    schema.add_relationship('aka_title', 'movie_id', 'title', 'id') #1
    #schema.add_relationship('aka_title', 'kind_id', 'kind_type', 'id')

    # kind_type, no relationships

    return schema


def gen_job_light_ranges_imdb_schema(csv_path):
    """
    Just like the full IMDB schema but without tables that are not used in the job-light benchmark.
    """

    schema = SchemaGraph()


    # tables

    # title
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id',
                                                'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
                                                'series_years', 'md5sum'],
                           irrelevant_attributes=['episode_of_id', 'title',
                                                  'imdb_id',  'md5sum'],
                           no_compression=['kind_id'],
                           csv_file_location=csv_path.format('title'),
                           table_size=3486660))

    # movie_info_idx
    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           irrelevant_attributes=['info', 'note'],
                           no_compression=['info_type_id'],
                           table_size=3147110))

    # movie_info
    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           csv_file_location=csv_path.format('movie_info'),
                           irrelevant_attributes=['info', 'note'],
                           no_compression=['info_type_id'],
                           table_size=24988000))

    # cast_info
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order',
                                                    'role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           irrelevant_attributes=['note', 'person_id', 'person_role_id'],
                           no_compression=['role_id'],
                           table_size=63475800))

    # movie_keyword
    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           csv_file_location=csv_path.format('movie_keyword'),
                           no_compression=['keyword_id'],
                           table_size=7522600))

    # movie_companies
    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           csv_file_location=csv_path.format('movie_companies'),
                           irrelevant_attributes=['note'],
                           no_compression=['company_id', 'company_type_id'],
                           table_size=4958300))

    # relationships

    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')
    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    return schema


def gen_job_union_schema(csv_path):
    """
    Specifies full imdb schema. Also tables not in the job-light benchmark.
    """
    schema = SchemaGraph()
    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'],
                           irrelevant_attributes=['episode_of_id', 'md5sum', 'season_nr', 'phonetic_code', 'imdb_index', 'imdb_id', 'series_years'],
                           csv_file_location=csv_path.format('title'),
                           no_compression=['kind_id'],
                           table_size=2528312))

    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           irrelevant_attributes=['note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           no_compression=['info_type_id'],
                           table_size=1380035))

    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_info'),
                           no_compression=['info_type_id'],
                           table_size=14835720))

    schema.add_table(Table('info_type', attributes=['id', 'info'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('info_type'),
                           table_size=113))


    schema.add_table(Table('complete_cast', attributes=['id', 'movie_id', 'subject_id', 'status_id'],
                           irrelevant_attributes=[ 'status_id'],
                           csv_file_location=csv_path.format('complete_cast'),
                           table_size=135086))
    
    
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                           irrelevant_attributes=['nr_order', 'person_id', 'person_role_id'],
                           csv_file_location=csv_path.format('cast_info'),
                           no_compression=['role_id'],
                           table_size=36244344))
        

    schema.add_table(Table('comp_cast_type', attributes=['id', 'kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('comp_cast_type'),
                           table_size=4))
        

    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_keyword'),
                           no_compression=['keyword_id'],
                           table_size=4523930))

    
    schema.add_table(Table('keyword', attributes=['id', 'keyword', 'phonetic_code'],
                           irrelevant_attributes=['phonetic_code'],
                           csv_file_location=csv_path.format('keyword'),
                           table_size=134170))
    


    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_companies'),
                           no_compression=['company_id', 'company_type_id'],
                           table_size=2609129))
    

    schema.add_table(Table('company_name', attributes=['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                           irrelevant_attributes=['name_pcode_sf', 'md5sum', 'name_pcode_nf', 'imdb_id'],
                           csv_file_location=csv_path.format('company_name'),
                           table_size=234997))


    schema.add_table(Table('company_type', attributes=['id', 'kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('company_type'),
                           table_size=4))


    schema.add_table(Table('aka_title', attributes=['id', 'movie_id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'note', 'md5sum'],
                           irrelevant_attributes=['episode_of_id', 'production_year', 'title', 'md5sum', 'season_nr', 'kind_id', 'note', 'phonetic_code', 'imdb_index', 'episode_nr'],
                           csv_file_location=csv_path.format('aka_title'),
                           table_size=361472))
    


    schema.add_table(Table('kind_type', attributes=['id','kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('kind_type'),
                           table_size=7))


    schema.add_table(Table('movie_link', attributes=['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
                           irrelevant_attributes=[ 'linked_movie_id'],
                           csv_file_location=csv_path.format('movie_link'),
                           table_size=29997))
    

    schema.add_table(Table('link_type', attributes=['id', 'link'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('link_type'),
                           table_size=18))

    
    schema.add_relationship('title', 'kind_id','kind_type', 'id') #in full

    
    schema.add_relationship('movie_info_idx', 'info_type_id','info_type','id') #XXX not in full
    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id') #in full

    schema.add_relationship('cast_info', 'movie_id', 'title', 'id') #in full

    schema.add_relationship('complete_cast', 'subject_id', 'comp_cast_type', 'id') #in full
    schema.add_relationship('complete_cast', 'movie_id', 'title', 'id') #in full
    
    schema.add_relationship('movie_keyword', 'keyword_id', 'keyword', 'id') #in full
    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id') #in full
        
    schema.add_relationship('movie_companies', 'company_id', 'company_name', 'id') #in full
    schema.add_relationship('movie_companies', 'company_type_id', 'company_type', 'id') #in full
    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id') #in full
    
    schema.add_relationship('aka_title', 'movie_id', 'title', 'id') #in full
    
    schema.add_relationship('movie_info', 'movie_id', 'title', 'id') #in full

    schema.add_relationship('movie_link', 'link_type_id', 'link_type', 'id') #in full
    schema.add_relationship('movie_link', 'movie_id', 'title', 'id')
    

    return schema


def gen_imdb_full_schema(csv_path):
    
    schema = SchemaGraph()
    
    schema.add_table(Table('name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('name'),
                           table_size=4167491))
    

    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_companies'),
                           no_compression=['company_id', 'company_type_id'],
                           table_size=2609129))
    

    schema.add_table(Table('aka_name', attributes=['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('aka_name'),
                           table_size=901343))
    

    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_info'),
                           no_compression=['info_type_id'],
                           table_size=14835720))
    

    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_keyword'),
                           no_compression=['keyword_id'],
                           table_size=4523930))
    

    schema.add_table(Table('person_info', attributes=['id', 'person_id', 'info_type_id', 'info', 'note'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('person_info'),
                           table_size=2963664))
    

    schema.add_table(Table('comp_cast_type', attributes=['id', 'kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('comp_cast_type'),
                           table_size=4))
    

    schema.add_table(Table('complete_cast', attributes=['id', 'movie_id', 'subject_id', 'status_id'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('complete_cast'),
                           table_size=135086))
    

    schema.add_table(Table('char_name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('char_name'),
                           table_size=3140339))
    

    schema.add_table(Table('movie_link', attributes=['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_link'),
                           table_size=29997))
    

    schema.add_table(Table('company_type', attributes=['id', 'kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('company_type'),
                           table_size=4))
    

    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('cast_info'),
                           no_compression=['role_id'],
                           table_size=36244344))
    

    schema.add_table(Table('info_type', attributes=['id', 'info'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('info_type'),
                           table_size=113))
    

    schema.add_table(Table('company_name', attributes=['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('company_name'),
                           table_size=234997))
    

    schema.add_table(Table('aka_title', attributes=['id', 'movie_id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'note', 'md5sum'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('aka_title'),
                           table_size=361472))
    

    schema.add_table(Table('kind_type', attributes=['id', 'kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('kind_type'),
                           table_size=7))
    

    schema.add_table(Table('role_type', attributes=['id', 'role'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('role_type'),
                           table_size=12))
    

    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           no_compression=['info_type_id'],
                           table_size=1380035))
    

    schema.add_table(Table('keyword', attributes=['id', 'keyword', 'phonetic_code'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('keyword'),
                           table_size=134170))
    

    schema.add_table(Table('link_type', attributes=['id', 'link'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('link_type'),
                           table_size=18))
    

    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'],
                           irrelevant_attributes=[],
                           no_compression=['kind_id'],
                           csv_file_location=csv_path.format('title'),
                           table_size=2528312))
    
    schema.add_relationship('aka_title', 'movie_id', 'title', 'id')

    schema.add_relationship('movie_link', 'movie_id', 'title', 'id')

    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')

    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')

    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')

    schema.add_relationship('title', 'kind_id', 'kind_type', 'id')

    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')

    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    schema.add_relationship('complete_cast', 'movie_id', 'title', 'id')

    schema.add_relationship('movie_link', 'link_type_id', 'link_type', 'id')

    schema.add_relationship('cast_info', 'person_role_id', 'char_name', 'id')

    schema.add_relationship('cast_info', 'role_id', 'role_type', 'id')

    schema.add_relationship('cast_info', 'person_id', 'name', 'id')

    schema.add_relationship('movie_info_idx', 'info_type_id', 'info_type', 'id')

    schema.add_relationship('movie_keyword', 'keyword_id', 'keyword', 'id')

    schema.add_relationship('movie_companies', 'company_id', 'company_name', 'id')

    schema.add_relationship('movie_companies', 'company_type_id', 'company_type', 'id')

    schema.add_relationship('complete_cast', 'subject_id', 'comp_cast_type', 'id')

    schema.add_relationship('aka_name', 'person_id', 'name', 'id')

    schema.add_relationship('person_info', 'person_id', 'name', 'id')
    
    return schema

def gen_job_original_schema(csv_path):
    
    schema = SchemaGraph()
        
    schema.add_table(Table('cast_info', attributes=['id', 'person_id', 'movie_id', 'person_role_id', 'note', 'nr_order', 'role_id'],
                           irrelevant_attributes=['nr_order'],
                           csv_file_location=csv_path.format('cast_info'),
                           table_size=36244344))
    

    schema.add_table(Table('char_name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           irrelevant_attributes=['imdb_index', 'surname_pcode', 'name_pcode_nf', 'imdb_id', 'md5sum'],
                           csv_file_location=csv_path.format('char_name'),
                           table_size=3140339))
    

    schema.add_table(Table('company_name', attributes=['id', 'name', 'country_code', 'imdb_id', 'name_pcode_nf', 'name_pcode_sf', 'md5sum'],
                           irrelevant_attributes=['name_pcode_nf', 'imdb_id', 'name_pcode_sf', 'md5sum'],
                           csv_file_location=csv_path.format('company_name'),
                           table_size=234997))
    

    schema.add_table(Table('company_type', attributes=['id', 'kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('company_type'),
                           table_size=4))
    

    schema.add_table(Table('movie_companies', attributes=['id', 'movie_id', 'company_id', 'company_type_id', 'note'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_companies'),
                           table_size=2609129))
    

    schema.add_table(Table('role_type', attributes=['id', 'role'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('role_type'),
                           table_size=12))
    

    schema.add_table(Table('title', attributes=['id', 'title', 'imdb_index', 'kind_id', 'production_year', 'imdb_id', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'series_years', 'md5sum'],
                           irrelevant_attributes=['imdb_index', 'phonetic_code', 'imdb_id', 'series_years', 'md5sum', 'episode_of_id', 'season_nr'],
                           csv_file_location=csv_path.format('title'),
                           table_size=2528312))
    

    schema.add_table(Table('keyword', attributes=['id', 'keyword', 'phonetic_code'],
                           irrelevant_attributes=['phonetic_code'],
                           csv_file_location=csv_path.format('keyword'),
                           table_size=134170))
    

    schema.add_table(Table('link_type', attributes=['id', 'link'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('link_type'),
                           table_size=18))
    

    schema.add_table(Table('movie_keyword', attributes=['id', 'movie_id', 'keyword_id'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_keyword'),
                           table_size=4523930))
    

    schema.add_table(Table('movie_link', attributes=['id', 'movie_id', 'linked_movie_id', 'link_type_id'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_link'),
                           table_size=29997))
    

    schema.add_table(Table('info_type', attributes=['id', 'info'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('info_type'),
                           table_size=113))
    

    schema.add_table(Table('movie_info', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('movie_info'),
                           table_size=14835720))
    

    schema.add_table(Table('movie_info_idx', attributes=['id', 'movie_id', 'info_type_id', 'info', 'note'],
                           irrelevant_attributes=['note'],
                           csv_file_location=csv_path.format('movie_info_idx'),
                           table_size=1380035))
    

    schema.add_table(Table('kind_type', attributes=['id', 'kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('kind_type'),
                           table_size=7))
    

    schema.add_table(Table('aka_title', attributes=['id', 'movie_id', 'title', 'imdb_index', 'kind_id', 'production_year', 'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr', 'note', 'md5sum'],
                           irrelevant_attributes=['episode_nr', 'imdb_index', 'note', 'phonetic_code', 'md5sum', 'episode_of_id', 'production_year', 'season_nr', 'kind_id'],
                           csv_file_location=csv_path.format('aka_title'),
                           table_size=361472))
    

    schema.add_table(Table('aka_name', attributes=['id', 'person_id', 'name', 'imdb_index', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           irrelevant_attributes=['imdb_index', 'surname_pcode', 'name_pcode_cf', 'name_pcode_nf', 'md5sum'],
                           csv_file_location=csv_path.format('aka_name'),
                           table_size=901343))
    

    schema.add_table(Table('name', attributes=['id', 'name', 'imdb_index', 'imdb_id', 'gender', 'name_pcode_cf', 'name_pcode_nf', 'surname_pcode', 'md5sum'],
                           irrelevant_attributes=['imdb_index', 'surname_pcode', 'name_pcode_nf', 'imdb_id', 'md5sum'],
                           csv_file_location=csv_path.format('name'),
                           table_size=4167491))
    

    schema.add_table(Table('comp_cast_type', attributes=['id', 'kind'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('comp_cast_type'),
                           table_size=4))
    

    schema.add_table(Table('complete_cast', attributes=['id', 'movie_id', 'subject_id', 'status_id'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('complete_cast'),
                           table_size=135086))
    

    schema.add_table(Table('person_info', attributes=['id', 'person_id', 'info_type_id', 'info', 'note'],
                           irrelevant_attributes=[],
                           csv_file_location=csv_path.format('person_info'),
                           table_size=2963664))

    schema.add_relationship('aka_title', 'movie_id', 'title', 'id')

    schema.add_relationship('movie_link', 'movie_id', 'title', 'id')

    schema.add_relationship('cast_info', 'movie_id', 'title', 'id')

    schema.add_relationship('movie_info', 'movie_id', 'title', 'id')

    schema.add_relationship('movie_info_idx', 'movie_id', 'title', 'id')

    schema.add_relationship('title', 'kind_id', 'kind_type', 'id')

    schema.add_relationship('movie_keyword', 'movie_id', 'title', 'id')

    schema.add_relationship('movie_companies', 'movie_id', 'title', 'id')

    schema.add_relationship('complete_cast', 'movie_id', 'title', 'id')

    schema.add_relationship('movie_link', 'link_type_id', 'link_type', 'id')

    schema.add_relationship('cast_info', 'person_role_id', 'char_name', 'id')

    schema.add_relationship('cast_info', 'role_id', 'role_type', 'id')

    schema.add_relationship('cast_info', 'person_id', 'name', 'id')

    schema.add_relationship('movie_info_idx', 'info_type_id', 'info_type', 'id')

    schema.add_relationship('movie_keyword', 'keyword_id', 'keyword', 'id')

    schema.add_relationship('movie_companies', 'company_id', 'company_name', 'id')

    schema.add_relationship('movie_companies', 'company_type_id', 'company_type', 'id')

    schema.add_relationship('complete_cast', 'subject_id', 'comp_cast_type', 'id')

    schema.add_relationship('aka_name', 'person_id', 'name', 'id')

    schema.add_relationship('person_info', 'person_id', 'name', 'id')
    
    return schema