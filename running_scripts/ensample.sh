export POLARS_MAX_THREADS=10

gpu_id=6
python_file=train.py

args="
--model ensamble
--fold_idx 2
--ensamble_models transformer lstm
--ensamble_ckpts '/data/shared_workspace/pengjie/NN-in-Finance/models/transformer/3model_best.pth' '/data/shared_workspace/pengjie/NN-in-Finance/models/lstm/3model_best.pth'
--manual_weights 1 1 1
"
CUDA_VISIBLE_DEVICES=$gpu_id python -u $python_file ${args}

# args="
# --model transformer"
# CUDA_VISIBLE_DEVICES=$gpu_id python -u $python_file ${args}

