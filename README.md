Technical Paper (link)



## Installation

1. Download Github repository

```bash
git clone https://github.com/postechdblab/learned-cardinality-estimation.git
```

2. Set up environments

```bash
# environment for FCN, MSCN, and FCN+Pool
bash environments/MSCN/setup.sh

# environment for E2E
bash environments/E2E/setup.sh

# environment for NeuroCard and UAE
bash environments/NeuroCard/setup.sh

# environment for DeepDB, DeepDB-JCT, and DeepDB-JCT-NARU
bash environments/DeepDB/setup.sh
```

3. Download datasets and unzip

```txt
# IMDB
# Original dataset: http://homepages.cwi.nl/~boncz/job/imdb.tgz
# We used Postgres exported version. The original dataset can make parsing errors in the Pandas parsing engine. 
link: https://drive.google.com/file/d/1j8DZd0TwZ6fBFN9FqXOzfYsFz_pDGiMp/view?usp=sharing
location: ./datasets/job

# TPC-DS
# Origianl dataset: http://tpc.org/tpc_documents_current_versions/current_specifications5.asp
# We used Postgres exported version. The original dataset can make parsing errors in the Pandas parsing engine. 
link: https://drive.google.com/file/d/1FIZjv6gsGFq74OXBGicusTfH1AxKRowu/view?usp=sharing
location: ./datasets/tpcds

# Synthetic
link: https://drive.google.com/file/d/1NrLysKrMIZ88Znnpm40HZKcYrMWI1iVF/view?usp=sharing
location: ./datasets/synthetic

# Sampled dataset (for bitmap features in query-driven methods)
link: https://drive.google.com/file/d/1xaJVLR9vcxsbW7Mx3eNVM2sF4PCTUQsN/view?usp=sharing
location: ./samples
```

4. Download models 

```txt
# FCN
link: https://drive.google.com/file/d/19bVvA8_Yj9tsTrLvlMagtZ68U5EvLoDr/view?usp=sharing
location: ./models/FCN

# MSCN
link: https://drive.google.com/file/d/1YRmodqlRFPkoqBcDaT7wiKi2orkm84hl/view?usp=sharing
location: ./models/MSCN

# FCN+Pool
link: https://drive.google.com/file/d/1JT7PSl8J0Jjqk29Dkamq8SIgVtxoIm1r/view?usp=sharing
location: ./models/FCN+Pool

# E2E
link: https://drive.google.com/file/d/1G6C5xIZQMLbRWLcCqF8c70wU_eRShOru/view?usp=sharing
location: ./models/E2E

# NeuroCard
link: https://drive.google.com/file/d/1lH1SpNJFj9eXHbCBc372mFnJFtYEOdoK/view?usp=sharing
location: ./models/NeuroCard

# UAE
link: https://drive.google.com/file/d/18vmBRTUwOE-z9p4oKolev4tlNygcy7fL/view?usp=sharing
location: ./models/UAE

# DeepDB(+JCT, +NARU)
link: https://drive.google.com/file/d/1aIuMcl9dp6uaZ7NYadbzM29S-xkwe7_t/view?usp=sharing
location: ./models/DeepDB
```

5. Download training queries and unzip

```txt
link: https://drive.google.com/file/d/1-O-fckKGuea09x5IQoANhQxAmKh5eRb2/view?usp=sharing
location: ./train
```

6. Download workloads and unzip

```txt
link: https://drive.google.com/file/d/1nyYk_fYg5uBe0wpu8b9l_OwTK0GDudUM/view?usp=sharing
location: ./workloads
```

7. Download pre-trained word embeddings and unzip

```txt
link: https://drive.google.com/file/d/10-RvdESO6Z4OtlLPZ4EqGrl22Am4Bemv/view?usp=sharing
location: ./wordvectors
```



## How to Run

1. Reproduce the FCN results in the paper

```bash
# Activate environment
source activate mscn
# Run experiment script
bash scripts/FCN.sh

# After executing script file, the result will be stored in ./results/FCN/<workload>_<database>.csv file.
```

2. Reproduce the MSCN results in the paper

```bash
# Activate environment
source activate mscn
# Run experiment script
bash scripts/MSCN.sh

# After executing script file, the result will be stored in ./results/MSCN/<workload>_<database>.csv file.
```

3. Reproduce the FCN+Pool results in the paper

```bash
# Activate environment
source activate mscn
# Run experiment script
bash scripts/FCN+Pool.sh

# After executing script file, the result will be stored in ./results/FCN+Pool/<workload>_<database>.csv file.
```

4. Reproduce the E2E results in the paper

```bash
# Activate environment
source activate e2e
# Run experiment script
bash scripts/E2E.sh

# After executing script file, the result will be stored in ./results/E2E/<workload>_<database>.csv file.
```

5. Reproduce the NeuroCard results in the paper

```bash
# Activate environment
source activate neurocard
# Run experiment script
bash scripts/NeuroCard.sh

# After executing script file, the result will be stored in ./results/NeuroCard/<workload>_<database>.csv file.
```

6. Reproduce the UAE results in the paper

```bash
# Activate environment
source activate neurocard
# Run experiment script
bash scripts/UAE.sh

# After executing script file, the result will be stored in ./results/UAE/<workload>_<database>.csv file.
```

7. Reproduce the DeepDB results in the paper

```bash
# Activate environment
source activate deepdb
# Run experiment script
bash scripts/DeepDB.sh

# After executing script file, the result will be stored in ./results/DeepDB/<workload>_<database>.csv file.
```

8. Reproduce the DeepDB-JCT results in the paper

```bash
# Activate environment
source activate deepdb
# Run experiment script
bash scripts/DeepDB-JCT.sh

# After executing script file, the result will be stored in ./results/DeepDB-JCT/<workload>_<database>.csv file.
```

9. Reproduce the DeepDB-JCT-NARU results in the paper

```bash
# Activate environment
source activate deepdb
# Run experiment script
bash scripts/DeepDB-JCT-NARU.sh

# After executing script file, the result will be stored in ./results/DeepDB-JCT-NARU/<workload>_<database>.csv file.
```



## Synthetic dataset index

### Syn-Single databases

1. Varying domain size: 

| Index | Domain size | Skewness | Correlation |
| ----- | ----------- | -------- | ----------- |
| 01    | 10          | 1.0      | 0.8         |
| 02    | 100         | 1.0      | 0.8         |
| 00    | 1k          | 1.0      | 0.8         |
| 23    | 10k         | 1.0      | 0.8         |

2. Varying skewness: 

| Index | Domain size | Skewness | Correlation |
| ----- | ----------- | -------- | ----------- |
| 03    | 1k          | 0.0      | 0.8         |
| 04    | 1k          | 0.2      | 0.8         |
| 05    | 1k          | 0.4      | 0.8         |
| 06    | 1k          | 0.6      | 0.8         |
| 07    | 1k          | 0.8      | 0.8         |
| 00    | 1k          | 1.0      | 0.8         |
| 18    | 1k          | 1.2      | 0.8         |
| 19    | 1k          | 1.4      | 0.8         |
| 20    | 1k          | 1.6      | 0.8         |
| 21    | 1k          | 1.8      | 0.8         |
| 22    | 1k          | 2.0      | 0.8         |

3. Varying correlation:

| Index | Domain size | Skewness | Correlation |
| ----- | ----------- | -------- | ----------- |
| 08    | 1k          | 1.0      | 0.0         |
| 09    | 1k          | 1.0      | 0.1         |
| 10    | 1k          | 1.0      | 0.2         |
| 11    | 1k          | 1.0      | 0.3         |
| 12    | 1k          | 1.0      | 0.4         |
| 13    | 1k          | 1.0      | 0.5         |
| 14    | 1k          | 1.0      | 0.6         |
| 15    | 1k          | 1.0      | 0.7         |
| 00    | 1k          | 1.0      | 0.8         |
| 16    | 1k          | 1.0      | 0.9         |
| 17    | 1k          | 1.0      | 1.0         |

 

### Syn-Multi databases

1. Varying domain size:

| Index | Fanout Domain Size | Fanout Skewness |
| ----- | ------------------ | --------------- |
| 02    | 10                 | 1.0             |
| 00    | 100                | 1.0             |
| 13    | 1k                 | 1.0             |

2. Varying skewness:

| Index | Fanout Domain Size | Fanout Skewness |
| ----- | ------------------ | --------------- |
| 03    | 100                | 0.0             |
| 04    | 100                | 0.2             |
| 05    | 100                | 0.4             |
| 06    | 100                | 0.6             |
| 07    | 100                | 0.8             |
| 00    | 100                | 1.0             |
| 08    | 100                | 1.20            |
| 14    | 100                | 1.24            |
| 15    | 100                | 1.28            |
| 16    | 100                | 1.32            |
| 17    | 100                | 1.36            |
| 09    | 100                | 1.40            |
| 18    | 100                | 1.44            |
| 19    | 100                | 1.48            |
| 20    | 100                | 1.52            |
| 21    | 100                | 1.56            |
| 10    | 100                | 1.6             |
| 11    | 100                | 1.8             |
| 12    | 100                | 2.0             |