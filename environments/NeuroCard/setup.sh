#!/bin/bash

sudo apt install build-essential
cd ./environments/NeuroCard

conda env create -f environment.yml

mkdir ../../results/NeuroCard -p
mkdir ../../results/UAE -p