import polars as pl
from dataset import StockDataset,DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm
from model import lstmMODEL,transMODEL, AveragingEnsemble, VotingEnsemble, BaggingEnsemble, get_models
from feature_engineer import *
import random
import argparse
import pathlib
import lightgbm as lgb
import time
import pickle

def parser_aug():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_index',type=int, default=2)
    parser.add_argument('--saveres_dir',type=str, default='./save_result/')
    parser.add_argument('--model', default="voting", choices=['transformer','lstm','lightgbm','averaging','voting','bagging'],type=str, help='type of model')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=1, type=int, help='Training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--root_path', default="data.csv", type=str, help='data_path')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
    parser.add_argument('--nvars', type=int, default=185, help='encoder input size')
    parser.add_argument('--hidden_size', default=60, type=int, help='hidden_size')
    parser.add_argument('--fold_idx', default=2, type=int, help='the time fold index')
    parser.add_argument('--ckpt_cache', type=str, default='models')
    parser.add_argument('--ensemble_models',nargs='+',type=str, default=['transformer', 'lstm', 'lightgbm'])
    parser.add_argument('--ensemble_num',type=int, default=3)
    parser.add_argument('--ensemble_ckpts',nargs='+',type=str, default=['model_ckpt/transformer.pth','model_ckpt/lstm.pth','model_ckpt/lightgbm.json'])
    parser.add_argument('--ensemble_style', type=str, default='manual_weight')
    parser.add_argument('--manual_weights',nargs='+',type=float, default=[1, 1, 1])
    parser.add_argument('--manual_style',nargs='+',type=str, default='auto')
    args = parser.parse_args()
    return args
args = parser_aug()

selected_gpu_index = args.gpu_index
torch.cuda.set_device(selected_gpu_index)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

INITIAL_DATA_RATIO = 0.4
FOLD_RATIO_STEP = 0.20
N_TIME_FOLD = 4

def set_random_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def r2(y_pred, y_true):
    return 1 - ((y_true - y_pred)**2).sum()/((y_true - y_true.mean())**2).sum()
def rmse_loss(predictions, targets):
    predictions = predictions.squeeze()
    mse = torch.mean((predictions - targets)**2)  # Compute Mean Squared Error
    rmse = torch.sqrt(mse+1e-6)  # Compute square root to get RMSE
    return rmse

def get_data_loader(train_data,test_data,batch_size,features):

    train_features = np.array(train_data[features], dtype=np.float32)
    train_label = np.array(train_data['y'], dtype=np.float32)

    test_features = np.array(test_data[features], dtype=np.float32)
    test_label = np.array(test_data['y'], dtype=np.float32)


    train_set = StockDataset(train_features,train_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = StockDataset(test_features,test_label)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader,test_loader


def eval_epoch(train_data, test_data, batch_size, epoch_idx, device, model, optimizer, features):
    _, test_loader = get_data_loader(train_data, test_data, batch_size, features)
    with torch.no_grad():
        final_targets = []
        final_outputs = []
        t_bar = tqdm(enumerate(test_loader))
        for i,batch in t_bar:

            features, labels = batch
            test_features,test_labels = features.to(device), labels.to(device)
            

            outputs = model(test_features)
            targets = test_labels.detach().cpu().numpy()
            output = outputs.detach().cpu().numpy()

            final_targets.append(targets)
            final_outputs.append(output)

            t_bar.set_description(f"Batch: {i}")

        final_targets = np.concatenate(final_targets).squeeze()
        final_outputs = np.concatenate(final_outputs).squeeze() 
        result = Tool.evalation(final_outputs,final_targets, 'Test')
    return result, model

def training_epoch(train_data, test_data, batch_size, epoch_idx, device, model, optimizer, features):
    train_loader, test_loader = get_data_loader(train_data, test_data, batch_size, features)
    train_targets = []
    train_outputs = []
    bar = tqdm(enumerate(train_loader))
    result = None
    for i, batch in bar:
        features, labels = batch
        train_features,train_labels = features.to(device), labels.to(device)
        train_features,train_labels = train_features.float(), train_labels.float()
        outputs = model(train_features)
        loss = rmse_loss(outputs.squeeze(), train_labels)
            

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1)
        optimizer.step()
    
        label = train_labels.detach().cpu().numpy()
        pred = outputs.detach().cpu().numpy()

        train_targets.append(label)
        train_outputs.append(pred)

        bar.set_description(f"epoch: {epoch_idx}, Batch: {i}, loss: {loss.detach().cpu().item():.4f}")
    train_targets = np.concatenate(train_targets).squeeze()
    train_outputs = np.concatenate(train_outputs).squeeze() 

    print("epoch:{}".format(epoch_idx+1))
    train_result = Tool.evalation(train_outputs, train_targets, 'Train')

    with torch.no_grad():
        final_targets = []
        final_outputs = []
        t_bar = tqdm(enumerate(test_loader))
        for i,batch in t_bar:

            features, labels = batch
            test_features,test_labels = features.to(device), labels.to(device)

            outputs = model(train_features)
            
            outputs = model(test_features)
            targets = test_labels.detach().cpu().numpy()
            output = outputs.detach().cpu().numpy()

            final_targets.append(targets)
            final_outputs.append(output)

            t_bar.set_description(f"Batch: {i}")
    
        final_targets = np.concatenate(final_targets).squeeze()
        final_outputs = np.concatenate(final_outputs).squeeze() 
        result = Tool.evalation(final_outputs,final_targets, 'Test')
    file_path = args.saveres_dir
    with open(file_path, "a") as file:
        file.write('Train: '+str(train_result)+'\n'+'Test: ' + str(result) + '\n')
    return result, model

def ensemble_predict(model, data_loader, device):
    preds = []
    data_labels = []
    for i, batch in tqdm(enumerate(data_loader)):
            features, labels = batch
            train_features,train_labels = features.to(device), labels.to(device)
            train_features,train_labels = train_features.float(), train_labels.float()
            outputs = model.predict(train_features)
            preds.append(outputs.detach().cpu().numpy())
            data_labels.append(train_labels.detach().cpu().numpy())
            # print(preds.shape)
    preds = np.concatenate(preds, axis=0)
    data_labels = np.concatenate(data_labels, axis=0)
    return preds, data_labels

def training_fold(fold_idx, data, augment_data, features, args, cache_dir):
    augment_names = ['rev_data', 'shift_10_data', 'shift_rev_10_data']
    s_time = time.time()
    whole_data_size = len(data)
    training_end_idx = int(whole_data_size * (INITIAL_DATA_RATIO + FOLD_RATIO_STEP * fold_idx)) - 10
    testing_end_idx = min(whole_data_size, int(whole_data_size * (INITIAL_DATA_RATIO + FOLD_RATIO_STEP * (fold_idx + 1))))
    train_data, test_data = data[: training_end_idx], data[training_end_idx: testing_end_idx]
    print(f"Time for data splitting: {time.time()- s_time}")
    # augment_training = augment_data[: training_end_idx]
    augment_training = {}
    for key in augment_data:
        augment_training[key] = augment_data[key][: training_end_idx]
    s_time = time.time()
    for feature in features:
        train_data, std, min_val, max_val, q001, q999 = normal_feature(train_data, feature)
        test_data = normal_test_feature(test_data, feature, std, min_val, max_val, q001, q999)
    for name in augment_names:
        for feature in features:
            augment_training[name], _, _, _, _, _ = normal_feature(augment_training[name], feature)
    print(f"Time for data normalization: {time.time()- s_time}")
    augment_training = data_set_concat(augment_training, augment_names, features+['y'])
    augment_training = pl.concat([train_data[features+['y']], augment_training])

    s_time = time.time()
    if args.model == 'lightgbm':
        new_data = (augment_training[features], augment_training['y'], test_data)
    else:
        new_data = (augment_training, test_data)

    if args.model == "transformer":
        model = transMODEL(args.nvars,args.hidden_size).to(DEVICE)
    elif args.model == "lstm" : 
        model = lstmMODEL(args.nvars,args.hidden_size).to(DEVICE)
    elif args.model == "lightgbm":
        model = lgb.LGBMRegressor(num_leaves=14, max_depth=4, n_jobs = 14)
    elif args.model == "averaging":
        model = AveragingEnsemble(args.ensemble_models, args.ensemble_ckpts, args, DEVICE, args.ensemble_style, args.manual_weights)
    elif args.model == 'voting':
        model = VotingEnsemble(args.ensemble_models, args.ensemble_num)
        model.set_optimizer(
        "Adam", 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay, 
        )
        model_list = get_models(args.ensemble_models, args.ensemble_ckpts, args, DEVICE)
        model.setting_models(model_list, rmse_loss)
        
        # loading
        model_weight_dict = torch.load(ckpt_model_path.joinpath('model_weight.pth'))
        model.load_state_dict(model_weight_dict, strict=False)
        
    elif args.model == 'bagging':
        model = BaggingEnsemble(args.ensemble_models, args.ensemble_num)
        model.set_optimizer(
        "Adam", 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay, 
        )
        model_list = get_models(args.ensemble_models, args.ensemble_ckpts, args, DEVICE)
        model.setting_models(model_list, rmse_loss)

    print(f"Time for model creation: {time.time()- s_time}")
    # path create
    ckpt_path = pathlib.Path(args.ckpt_cache)
    ckpt_path.mkdir(exist_ok=True)

    ckpt_model_path = ckpt_path.joinpath(args.model)
    ckpt_model_path.mkdir(exist_ok=True)

    if args.model == 'lightgbm':
        print('Training lightgbm')
        gbm = model.fit(new_data[0], new_data[1])
        
        gbm.booster_.save_model(ckpt_model_path.joinpath('model_lgb.json'))
        train_pred = model.predict(new_data[0])
        test_pred = model.predict(new_data[2][features])

        Tool.evalation(train_pred, new_data[1], "Train")
        result = Tool.evalation(test_pred, new_data[2]['y'], "Test")
        return result
    elif args.model == 'voting':
        train_loader, test_loader = get_data_loader(new_data[0], new_data[1], args.batch_size, features)
        model.fit(train_loader, args.epochs)
        # torch.save(model, ckpt_model_path.joinpath("model_best.pth"))
        dump_dict = {}
        with torch.no_grad():
            for pn, p in model.named_parameters():
                if "model_weight" in pn:
                    dump_dict[pn] = p
        torch.save(dump_dict, ckpt_model_path.joinpath('model_weight.pth'))
        
        train_pred, train_label = ensemble_predict(model, train_loader, DEVICE)
        test_pred, test_label = ensemble_predict(model, test_loader, DEVICE)

        train_result = Tool.evalation(train_pred, train_label, "Train")
        result = Tool.evalation(test_pred, test_label, "Test")
        file_path = args.saveres_dir
        with open(file_path, "a") as file:
            file.write('Train: '+str(train_result)+'\n'+'Test: ' + str(result) + '\n')
        return result
    elif args.model == 'bagging':
        train_loader, test_loader = get_data_loader(new_data[0], new_data[1], args.batch_size, features)
        model.fit(train_loader, args.epochs)
        dump_dict = {}
        with torch.no_grad():
            for pn, p in model.named_parameters():
                if "model_weight" in pn:
                    dump_dict[pn] = p
        torch.save(dump_dict, ckpt_model_path.joinpath('model_weight.pth'))
        
        train_pred, train_label = ensemble_predict(model, train_loader, DEVICE)
        test_pred, test_label = ensemble_predict(model, test_loader, DEVICE)

        train_result = Tool.evalation(train_pred, train_label, "Train")
        result = Tool.evalation(test_pred, test_label, "Test")
        file_path = args.saveres_dir
        with open(file_path, "a") as file:
            file.write('Train: '+str(train_result)+'\n'+'Test: ' + str(result) + '\n')
        return result
    else:
        if args.manual_style == 'manual':
            eval_epoch(new_data[0], new_data[1], args.batch_size, epoch, DEVICE, model, optimizer, features)
        else:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
            best_result = None
            
            for epoch in range(args.epochs):
                result, model = training_epoch(
                    new_data[0], new_data[1], args.batch_size, epoch, DEVICE, model, optimizer, features
                )
                if best_result is None:
                    best_result = result
                    torch.save(model.state_dict(), ckpt_model_path.joinpath("model_best.pth"))
                elif result['R2'] > best_result['R2']:
                    torch.save(model.state_dict(), ckpt_model_path.joinpath("model_best.pth"))
                    best_result = result
            return best_result



if __name__ == "__main__":

    args = parser_aug()
    s_time = time.time()
    set_random_seed(1)
    data = pl.read_csv('data.csv')
    cache_dir = pathlib.Path('cache')
    augment_data, aug_features, aug_features_flat = get_augment_data(data)

    if cache_dir.joinpath('get_features_cache.pkl').exists():
        with open(cache_dir.joinpath('get_features_cache.pkl'), 'rb') as fin:
            data, features, art_targets = pickle.load(fin)
    else:
        with open(cache_dir.joinpath('get_features_cache.pkl'), 'wb') as fin:
            data, features, art_targets = get_features(data)
            pickle.dump((data, features, art_targets), fin)

    print(f"Loading data: {time.time()-s_time}")
    training_fold(args.fold_idx, data, augment_data, features, args, cache_dir)

