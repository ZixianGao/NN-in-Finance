#!/bin/bash
###
 # @Author: pengjie pengjieb@mail.ustc.edu.cn
 # @Date: 2024-03-24 00:27:10
 # @LastEditors: pengjie pengjieb@mail.ustc.edu.cn
 # @LastEditTime: 2024-03-24 01:31:35
 # @FilePath: /NN-in-Finance/shells/eval_voting.sh
 # @Description: 
 # 
 # Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
### 

export POLARS_MAX_THREADS=10



gpu_id=0
python_file=evaluation.py
model="ensemble"
data_path="testing_data.csv"
model_cache="model_ckpt"
output_path="output.csv"
data_cache="cache"
batch_size=64
if_label=False
random_seed=1

args="
--model $model 
--batch_size $batch_size 
--if_label $if_label 
--data_path $data_path 
--model_cache $model_cache 
--output_path $output_path 
--data_cache $data_cache 
--random_seed $random_seed
--ensamble_ckpts models/lightgbm/#model_lgb3.json models/lstm/3model_best.pth models/transformer/3model_best.pth
--ensamble_models lightgbm lstm transformer"

CUDA_VISIBLE_DEVICES=$gpu_id python -u $python_file ${args}

read -p "Press [Enter] key to continue..."
