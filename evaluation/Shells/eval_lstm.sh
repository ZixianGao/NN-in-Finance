#!/bin/bash

export POLARS_MAX_THREADS=10

gpu_id=0
python_file=train.py
model="lstm"
data_path="data.csv"
model_cache="model_ckpt"
output_path="output.csv"
data_cache="cache"
batch_size=64
if_label=False
random_seed=1

args="
--model $model --batch_size $batch_size --if_label $if_label --data_path $data_path --model_cache $model_cache --output_path $output_path --data_cache $data_cache --random_seed $random_seed"

CUDA_VISIBLE_DEVICES=$gpu_id python -u $python_file ${args}

read -p "Press [Enter] key to continue..."