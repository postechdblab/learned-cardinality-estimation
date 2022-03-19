import argparse
import os
import json
import re
from src.feature_extraction.database_loader import load_dataset
import pandas as pd
import re

from src.feature_extraction.database_loader_new import load_dataset
from src.feature_extraction.extract_features import add_sample_bitmap_wo_data, feature_extractor, add_sample_bitmap

NUM_SAMPLES=1000

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to input query json file", type = str)
    parser.add_argument("--sample", help="path to sample directory", type = str)

    args = parser.parse_args()

    input_path = args.input
    seq_path = input_path.replace(".json", "_seq.json")
    output_path = input_path.replace(".json", "_seq_sample.json")
    
    print("load samples")
    _, sample = load_dataset(args.sample)
    
    print("extract feature")
    feature_extractor(input_path, seq_path)
    print("add sample bitmaps")
    add_sample_bitmap_wo_data(seq_path, output_path, sample, NUM_SAMPLES)

if __name__ == "__main__":
    main()
