export POLARS_MAX_THREADS=10

gpu_id=7
python_file=train.py

args="
--model lightgbm
--fold_idx 0"
CUDA_VISIBLE_DEVICES=$gpu_id python -u $python_file ${args}

args="
--model transformer"
CUDA_VISIBLE_DEVICES=$gpu_id python -u $python_file ${args}
