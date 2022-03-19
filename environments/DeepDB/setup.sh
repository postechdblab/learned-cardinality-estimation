#!/bin/bash
sudo apt install -y build-essential libpq-dev gcc python3-dev 

cd ./environments/DeepDB
conda create -y -n deepdb python=3.7.7
. activate deepdb

python3 -m pip install -r ./requirements.txt

mkdir ../../results/DeepDB -p
mkdir ../../results/DeepDB-JCT -p
mkdir ../../results/DeepDB-JCT-NARU -p
