#!/usr/bin/bash

property=$1
connectivity_type=$2
target_dir='data/descriptors/'$property'/'$connectivity_type

find $target_dir -maxdepth 1 -type f | while read FILE; do
  python script/normalization.py \
                --property $property \
                --file_path $FILE
done
