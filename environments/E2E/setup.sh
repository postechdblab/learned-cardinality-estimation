#!/bin/bash

pushd ./environments/E2E
conda env create -f environment.yml

popd
mkdir ./results/E2E -p
