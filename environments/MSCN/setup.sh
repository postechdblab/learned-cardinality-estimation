#!/bin/bash

pushd ./environments/MSCN
conda env create -f environment.yml

popd
mkdir ./results/MSCN -p
mkdir ./results/FCN -p
mkdir ./results/FCN+Pool -p
