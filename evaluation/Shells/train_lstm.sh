export POLARS_MAX_THREADS=10
gpu_id=0
python_file=train.py
model="lstm"
args="
--model $model --gpu_index $gpu_id --saveres_dir "./save_result/"$model"_result.txt" --learning_rate 1e-5 "
CUDA_VISIBLE_DEVICES=$gpu_id python -u $python_file ${args}