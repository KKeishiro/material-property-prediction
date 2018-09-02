#!/usr/bin/bash

property=$1
connectivity=$2
descriptor_type=$3
cov_radii_tol=0.65
save_dir='../data/descriptors/'

if [ $connectivity = 'distance' ]; then
  for i in 'element_wise' 'matmul'; do
    if [ $descriptor_type = 'structure' ]; then
      for j in 'mean' 'trace' 'std'; do
        python script/graph_descriptors_joblib.py \
                --property $property \
                --cov_radii_tol $cov_radii_tol \
                --multiply_type $i \
                --descriptor_type $descriptor_type \
                --quantize_strc $j \
                --save_path $save_dir$property'/'$connectivity'/X_'$i'_'$j'_strc.csv' \
                --periodic
      done

    elif [ $descriptor_type = 'mixtured' ]; then
      for j in 'sum' 'mean' 'trace'; do
        python script/graph_descriptors_joblib.py \
                --property $property \
                --cov_radii_tol $cov_radii_tol \
                --multiply_type $i \
                --descriptor_type $descriptor_type \
                --quantize_mix $j \
                --save_path $save_dir$property'/'$connectivity'/X_'$i'_'$j'_mixtured.csv' \
                --periodic
      done
    fi
  done

elif [ $connectivity = 'weight' ]; then
  for i in 'element_wise' 'matmul'; do
    if [ $descriptor_type = 'structure' ]; then
      for j in 'mean' 'trace' 'std'; do
        python script/graph_descriptors_joblib.py \
                --property $property \
                --connectivity $connectivity \
                --multiply_type $i \
                --descriptor_type $descriptor_type \
                --quantize_strc $j \
                --save_path $save_dir$property'/'$connectivity'/X_'$i'_'$j'_strc.csv' \
                --periodic
      done

    elif [ $descriptor_type = 'mixtured' ]; then
      for j in 'sum' 'mean' 'trace'; do
        python script/graph_descriptors_joblib.py \
                --property $property \
                --connectivity $connectivity \
                --multiply_type $i \
                --descriptor_type $descriptor_type \
                --quantize_mix $j \
                --save_path $save_dir$property'/'$connectivity'/X_'$i'_'$j'_mixtured.csv' \
                --periodic
      done
    fi
  done
fi
