import argparse
import pathlib
import pickle
import random
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset import EvalDataset
from feature_engineer import *
from model import EnsambleModel, loading_lstm, loading_transformer

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_data(data, cache_dir, label=False):
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    if cache_dir.joinpath('get_features_cache.pkl').exists():
        print('features cache exist')
        with open(cache_dir.joinpath('get_features_cache.pkl'), 'rb') as fin:
            data, features, art_targets = pickle.load(fin)
    else:
        print('features cache do not exist, creating features cache')
        with open(cache_dir.joinpath('get_features_cache.pkl'), 'wb') as fin:
            data, features, art_targets = get_features(data)
            pickle.dump((data, features, art_targets), fin)

    for feature in features:
        data = normal_test_feature(data, feature)

    if label:
        label_data = data['y']
    else:
        label_data = None

    return data[features], features, label_data

def get_data_loader(test_data, batch_size, label=None):
    test_features = np.array(test_data, dtype=np.float32)
    if label is not None:
        test_label = np.array(label, dtype=np.float32)
        test_set = EvalDataset(test_features, test_label)
    else:
        test_set = EvalDataset(test_features)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return test_loader

def eval_dl_model(test_data, model, args, label=None):
    test_loader = get_data_loader(test_data, args.batch_size, label)
    with torch.no_grad():
        final_targets = []
        final_outputs = []
        t_bar = tqdm(enumerate(test_loader))
        for i, batch in t_bar:
            if label is not None:
                features, labels = batch
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                targets = labels.detach().cpu().numpy()
                final_targets.append(targets)
            else:
                features = batch
                features = features.to(DEVICE)

            outputs = model(features)
            output = outputs.detach().cpu().numpy()
            final_outputs.append(output)

            t_bar.set_description(f"Batch: {i}")

        final_outputs = np.concatenate(final_outputs).squeeze()

        if label is not None:
            final_targets = np.concatenate(final_targets).squeeze()
            result = Tool.evalation(final_outputs, final_targets, 'Test')
            return result
        else:
            rows, cols = final_outputs.shape
            result_length = rows + cols - 1
            result = np.zeros(result_length)
            count = np.zeros(result_length)

            for i in range(rows):
                for j in range(cols):
                    result_idx = i + j
                    result[result_idx] += final_outputs[i, j]
                    count[result_idx] += 1

            return result / count


def eval_model(test_data, features, args, cache_dir, label=None):
    if args.model == "transformer":
        file_name = pathlib.Path(cache_dir).joinpath("transformer.pth")
        model = loading_transformer(file_name=file_name, args=args).to(DEVICE)
    elif args.model == "lstm":
        file_name = pathlib.Path(cache_dir).joinpath("lstm.pth")
        model = loading_lstm(file_name=file_name, args=args).to(DEVICE)
    elif args.model == "lightgbm":
        file_name = pathlib.Path(cache_dir).joinpath("lightgbm.json")
        model = lgb.Booster(model_file=file_name)
    elif args.model == "ensemble":
        model = EnsambleModel(args.ensamble_models, args.ensamble_ckpts, args, DEVICE, args.ensamble_style, args.manual_weights)
    else:
        raise AssertionError(f"Unsupported model type: {args.model}")

    if args.model == 'lightgbm':
        test_pred = model.predict(test_data[features])
        if label is not None:
            result = Tool.evalation(test_pred, label, "Test")
            return result
        else:
            return test_pred
    else:
        result = eval_dl_model(test_data, model, args, label)
        return result

def set_random_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def parser_aug():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--model', default="transformer",
                        choices=['transformer', 'lstm', 'lightgbm', 'ensemble'], type=str,
                        help='type of model')
    parser.add_argument('--ensemble_models', nargs='+', type=str, default=['transformers', 'lstm'])
    parser.add_argument('--ensemble_ckpts', nargs='+', type=str, default=[])
    parser.add_argument('--ensemble_style', type=str, default='manual')
    parser.add_argument('--nvars', type=int, default=185, help='encoder input size')
    parser.add_argument('--hidden_size', default=60, type=int, help='hidden_size')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--if_label', default=False, type=bool, help='whether have label and get r2 score')
    parser.add_argument('--data_path', default='data.csv', type=str, help='Path to the data file')
    parser.add_argument('--model_cache', default='model_ckpt', type=str, help='Path to the model checkpoint directory')
    parser.add_argument('--output_path', default='output.csv', type=str, help='Path to the output CSV file')
    parser.add_argument('--data_cache', default='cache', type=str, help='Path to the saved features')
    parser.add_argument('--random_seed', default=1, type=int, help='Random seed for reproducibility')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parser_aug()

    data = pl.read_csv(args.data_path)
    s_time = time.time()
    set_random_seed(args.random_seed)

    test_data, features, label = get_data(data=data, cache_dir=args.data_cache, label=args.if_label)
    print(f"Loading data: {(time.time() - s_time):.4f} seconds")

    s_time = time.time()
    result = eval_model(test_data=test_data, features=features, args=args, cache_dir=args.model_cache, label=label)
    print(f"Evaluating: {(time.time() - s_time):.4f} seconds")

    if not args.if_label:
        df = pd.DataFrame(result, columns=['pred'])
        df.to_csv(args.output_path, index=False)
        print(f"Saved results to {args.output_path}")
