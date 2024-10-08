import polars as pl
from dataset import StockDataset,DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from tqdm import tqdm
from TCnet_model.ModernTCN import ModernTCN
from model import lstmMODEL,transMODEL,rnnMODEL,S4Model,mlp
from feature_engineer import *
import random
import argparse
DEVICE = torch.device("cuda:3")

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--task_name', default="classification", type=str, help='type of task')
parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
parser.add_argument('--dims', nargs='+',type=int, default=[32,64], help='dmodels in each stage')#32 64
parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256], help='dw dims in dw conv in each stage')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--freq', type=str, default='h')
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--target_window', type=int, default=96, help='prediction sequence length')
parser.add_argument('--class_num', type=int, default=1, help='number of classe')
parser.add_argument('--model', default="lightgbm", choices=['transformer','lstm','rnn','S4',"TCN",'lightgbm','mlp'],type=str, help='type of model')
parser.add_argument('--learning_rate', default=1e-5, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay')
parser.add_argument('--epochs', default=10, type=int, help='Training epochs')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--n_layers', default=1, type=int, help='Number of layers')
# parser.add_argument('--model', default="rnn", choices=['transformer','lstm','rnn','S4'],type=str, help='type of model')
parser.add_argument('--input_feature_size', default=1, type=int, help='input_feature_size')
parser.add_argument('--input_size_mlp', default=185, type=int, help='input_size_mlp')
parser.add_argument('--root_path', default="data.csv", type=str, help='data_path')
parser.add_argument('--ffn_ratio', default=1, type=int)
parser.add_argument('--patch_size', default=8, type=int)
parser.add_argument('--patch_stride', default=4, type=int)
parser.add_argument('--num_blocks', default=[1,1], type=int)
parser.add_argument('--large_size', default=[13,13], type=int)
parser.add_argument('--small_size', default=[5,5], type=int)
# parser.add_argument('--dims', default=32, type=int)
parser.add_argument('--head_dropout', default=0.0, type=float)
parser.add_argument('--class_dropout', default=0.0, type=float)
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--itr', default=1, type=int)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--dex', default="Exp", type=str)
parser.add_argument('--use_multi_scale', default=False, type=bool)
parser.add_argument('--nvars', type=int, default=185, help='encoder input size')
parser.add_argument('--small_kernel_merged', type=bool, default=False, help='small_kernel has already merged or not')
parser.add_argument('--hidden_size', default=60, type=int, help='hidden_size')
args = parser.parse_args()

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
    mse = torch.mean((predictions - targets)**2)  # Compute Mean Squared Error
    rmse = torch.sqrt(mse+1e-6)  # Compute square root to get RMSE
    return rmse

def get_data_loader(train_data,test_data,batch_size,features):


    # Predict y
    # train_data, temp_data = train_test_split(data, test_size=split_size, shuffle=False)
    # valid_data, test_data = train_test_split(
        # temp_data, test_size=0.5, shuffle=False)

    train_features = np.array(train_data[features], dtype=np.float32)
    train_label = np.array(train_data['y'], dtype=np.float32)

    test_features = np.array(test_data[features], dtype=np.float32)
    test_label = np.array(test_data['y'], dtype=np.float32)

    # train_features = train_data[features]
    # train_label = train_data['y']

    # test_features = test_data[features]
    # test_label = test_data['y']

    train_set = StockDataset(train_features,train_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = StockDataset(test_features,test_label)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader,test_loader

def get_four_stage_dataloader(data,agument_data,batch_size,features,augfeatures):
    augment_names = ['rev_data', 'shift_10_data', 'shift_rev_10_data']

    #stage1:
    train_temp_data1, _ = train_test_split(data, test_size=0.45, shuffle=False)
    train_data_1, test_data_1 = train_test_split(train_temp_data1, test_size=0.15, shuffle=False)
    augment_temp_data1,_ = data_set_split (agument_data,augment_names,0.45)
    augment_train_data1,_ = data_set_split (augment_temp_data1,augment_names,0.15) 
    for feature in features:
        train_data_1, std1, min1, max1,q001_1,q999_1 = normal_feature(train_data_1, feature)
        test_data_1 = normal_test_feature(test_data_1, feature,  std1, min1, max1,q001_1,q999_1)
    for name in augment_names:
        for feature in features:
            augment_train_data1[name], _,_,_,_,_ = normal_feature(augment_train_data1[name], feature)
    augment_train_data1 = data_set_concat(augment_train_data1, augment_names, features+['y'])

    augment_train_data1 = pl.concat([train_data_1[features+['y']], augment_train_data1])
    if args.model == 'lightgbm':
        new_train_data1 = augment_train_data1[features]
        new_train_target1 = augment_train_data1['y']
        new_test_data1 = test_data_1
    else:
    # train_data_1,test_data_1 = test_normalize(train_data_1, test_data_1,features)
        train_loader1,test_loader1 = get_data_loader(augment_train_data1, test_data_1,batch_size,features)
    del train_temp_data1, augment_temp_data1, augment_train_data1,train_data_1,test_data_1

    #stage2:
    train_temp_data2, _ = train_test_split(data, test_size=0.3, shuffle=False)
    train_data_2, test_data_2 = train_test_split(train_temp_data2, test_size=0.15, shuffle=False)
    augment_temp_data2,_ = data_set_split (agument_data,augment_names,0.3)
    augment_train_data2,_ = data_set_split (augment_temp_data2,augment_names,0.15)   
    for feature in features:
        train_data_2, std2, min2, max2,q001_2,q999_2  = normal_feature(train_data_2, feature)
        test_data_2 = normal_test_feature(test_data_2, feature,  std2, min2, max2,q001_2,q999_2 )

    for name in augment_names:
        for feature in features:
            augment_train_data2[name], _,_,_,_,_ = normal_feature(augment_train_data2[name], feature)
    augment_train_data2 = data_set_concat(augment_train_data2, augment_names, features+['y'])
    augment_train_data2 = pl.concat([train_data_2[features+['y']], augment_train_data2])
    if args.model == 'lightgbm':
        new_train_data2 = augment_train_data2[features]
        new_train_target2 = augment_train_data2['y']
        new_test_data2 = test_data_2
    else:
    # train_data_1,test_data_1 = test_normalize(train_data_1, test_data_1,features)
        train_loader2,test_loader2 = get_data_loader(augment_train_data2, test_data_2,batch_size,features)
    del train_temp_data2, augment_temp_data2, augment_train_data2,train_data_2,test_data_2

    #stage3:
    train_temp_data3, _ = train_test_split(data, test_size=0.15, shuffle=False)
    train_data_3, test_data_3 = train_test_split(train_temp_data3, test_size=0.15, shuffle=False)
    augment_temp_data3,_ = data_set_split (agument_data,augment_names,0.15)
    augment_train_data3,_ = data_set_split (augment_temp_data3,augment_names,0.15)   
    for feature in features:
        train_data_3, std3, min3, max3,q001_3,q999_3  = normal_feature(train_data_3, feature)
        test_data_3 = normal_test_feature(test_data_3, feature,  std3, min3, max3,q001_3,q999_3 )

    for name in augment_names:
        for feature in features:
            augment_train_data3[name], _,_,_,_,_ = normal_feature(augment_train_data3[name], feature)
    augment_train_data3 = data_set_concat(augment_train_data3, augment_names, features+['y'])
    augment_train_data3 = pl.concat([train_data_3[features+['y']], augment_train_data3])
    if args.model == 'lightgbm':
        new_train_data3 = augment_train_data3[features]
        new_train_target3 = augment_train_data3['y']
        new_test_data3 = test_data_3
    else:
    # train_data_1,test_data_1 = test_normalize(train_data_1, test_data_1,features)
        train_loader3,test_loader3 = get_data_loader(augment_train_data3, test_data_3,batch_size,features)
    del train_temp_data3, augment_temp_data3, augment_train_data3,train_data_3,test_data_3

    #stage4:
    train_data_4,test_data_4 = train_test_split(data, test_size=0.15, shuffle=False)
    augment_train_data4,_= data_set_split (augment_data,augment_names,0.15) 

    for feature in features:
        train_data_4, std4, min4, max4,q001_4,q999_4  = normal_feature(train_data_4, feature)
    
        test_data_4 = normal_test_feature(test_data_4, feature,  std4, min4, max4,q001_4,q999_4 )

    for name in augment_names:
        for feature in features:
            augment_train_data4[name], _,_,_,_,_ = normal_feature(augment_train_data4[name], feature)   
    augment_train_data4 = data_set_concat(augment_train_data4, augment_names, features+['y'])
    augment_train_data4 = pl.concat([train_data_4[features+['y']], augment_train_data4])
    if args.model == 'lightgbm':
        new_train_data4 = augment_train_data4[features]
        new_train_target4 = augment_train_data4['y']
        new_test_data4 = test_data_4
    else:
    # train_data_1,test_data_1 = test_normalize(train_data_1, test_data_1,features)
        train_loader4,test_loader4 = get_data_loader(augment_train_data4, test_data_4,batch_size,features)
    del augment_train_data4,train_data_4,test_data_4
    if args.model == 'lightgbm':
        return new_train_data1,new_train_data2,new_train_data3,new_train_data4,new_train_target1,new_train_target2,new_train_target3,new_train_target4,new_test_data1,new_test_data2,new_test_data3,new_test_data4
    return train_loader1,test_loader1,train_loader2,test_loader2,train_loader3,test_loader3,train_loader4,test_loader4

if __name__ == "__main__":
    set_random_seed(1)
    data = pl.read_csv('data.csv')
    augment_data,aug_features,aug_features_flat = get_augment_data(data)
    data, features, art_targets = get_features(data)
    models = []
    if args.model == "lightgbm":
        new_train_data1,new_train_data2,new_train_data3,new_train_data4,new_train_target1,new_train_target2,new_train_target3,new_train_target4,new_test_data1,new_test_data2,new_test_data3,new_test_data4 = get_four_stage_dataloader(data,agument_data=augment_data,batch_size=args.batch_size,features=features,augfeatures=aug_features_flat)
        lgb_train_data = [new_train_data1,new_train_data2,new_train_data3,new_train_data4]
        lgb_train_target = [new_train_target1,new_train_target2,new_train_target3,new_train_target4]
        lgb_test = [new_test_data1,new_test_data2,new_test_data3,new_test_data4]
        model = lgb.LGBMRegressor(num_leaves=14, max_depth=4)
        models.append(model)
        for i in range(4):
            new_train_data = lgb_train_data[i]
            new_train_target = lgb_train_target[i]
            test_data = lgb_test[i]
            gbm = model.fit(new_train_data, new_train_target)
            gbm.save_model(f'#model_lgb{i+1}.json')
            # train_pred = model.predict(new_train_data)
            train_pred = model.predict(new_train_data)
            test_pred = model.predict(test_data[features])
            print(f"Stage {i+1}")
            #Tool.evalation(train_pred, new_train_target, "Train")
            Tool.evalation(train_pred, new_train_target, "Train")
            Tool.evalation(test_pred, test_data['y'], "Test")
    
    else:
        #print("111111111")
        train_data_loader_1,test_data_loader_1, train_data_loader_2, test_data_loader_2, train_data_loader_3,test_data_loader_3, train_data_loader_4,test_data_loader_4=get_four_stage_dataloader(data,agument_data=augment_data,batch_size=args.batch_size,features=features,augfeatures=aug_features_flat)
        #print("222222222")
        train_dataloaders = [train_data_loader_1, train_data_loader_2, train_data_loader_3, train_data_loader_4]
        test_dataloaders = [test_data_loader_1, test_data_loader_2, test_data_loader_3, test_data_loader_4]
        if args.model == "transformer":
            for i in range(4):
                model = transMODEL(args.nvars,args.hidden_size).to(DEVICE)
                models.append(model)
        elif args.model == "rnn" : 
            for i in range(4):
                model = rnnMODEL(args.nvars,args.hidden_size).to(DEVICE)
                models.append(model)

        elif args.model == "lstm" : 
            for i in range(4):
                print("11111",DEVICE)
                model = lstmMODEL(args.nvars,args.hidden_size).to(DEVICE)
                print("22222",DEVICE)
                models.append(model)
        
        elif args.model == "S4":
            for i in range(4):
                model = S4Model(d_input=args.nvars,
                        d_output=1,
                        d_model=args.hidden_size,
                        n_layers=args.n_layers,
                        dropout=args.dropout,
                        prenorm=False,
                        ) # 生成模型
                models.append(model)  
        elif args.model == "mlp":
            for i in range(4):
                model = mlp(input_size=args.input_size_mlp,
                            hidden_size=args.hidden_size, 
                            output_size=1, 
                            drop=args.dropout
                        ) # 生成模型
                models.append(model)  
                
        optimizers = []  # Create a list to store optimizers
        for model in models:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)  # Initialize optimizer for each model
            optimizers.append(optimizer)  # Append the optimizer to the list


        # 训练模型
        for model_index,(train_data_loader,test_data_loader) in enumerate(zip(train_dataloaders,test_dataloaders)):
            print(f'stage{model_index}---------------------------------------------------')
            model = models[model_index].to(DEVICE)
            # for name, param in model.named_parameters():
            #     print(f"Parameter name: {name}, dtype: {param.dtype}")
            # model = model.float()
            optimizer=optimizers[model_index]
            best_r2 =0

            for epoch in range(args.epochs):
                train_targets = []
                train_outputs = []
                # for i,batch in enumerate(tqdm(train_data_loader)):
                for i,batch in enumerate(train_data_loader):
                    features, labels = batch
                    train_features,train_labels = features.to(DEVICE),labels.to(DEVICE)
                    train_features,train_labels = train_features.float(),train_labels.float()

                # 将数据输入模型并计算损失

                    outputs = model(train_features)
                    loss = rmse_loss(outputs.squeeze(), train_labels)
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                

                    label = train_labels.detach().cpu().numpy()
                    pred = outputs.detach().cpu().numpy()

                    train_targets.append(label)
                    train_outputs.append(pred)
                # 打印每轮训练的损失
                train_targets = np.concatenate(train_targets).squeeze()
                train_outputs = np.concatenate(train_outputs).squeeze() 
                print("epoch:{}".format(epoch+1))
                Tool.evalation(train_outputs, train_targets, 'Train')

                with torch.no_grad():
                    final_targets = []
                    final_outputs = []
                    for i,batch in enumerate(test_data_loader):
                        # 将特征数据和标签数据转换为张量，并移动到模型设备上
                        features, labels = batch
                        test_features,test_labels = features.to(DEVICE),labels.to(DEVICE)
                # 将数据输入模型并计算损失
                        outputs = model(train_features)
                        
                        # 前向传播
                        outputs = model(test_features)
                        targets = test_labels.detach().cpu().numpy()
                        output = outputs.detach().cpu().numpy()

                        final_targets.append(targets)
                        final_outputs.append(output)

                    final_targets = np.concatenate(final_targets).squeeze()
                    final_outputs = np.concatenate(final_outputs).squeeze() 
                    Tool.evalation(final_outputs,final_targets, 'Test')
                    r_2 = r2(final_outputs,final_targets)
                    if r_2 > best_r2: 
                        best_r2 = r_2
                        torch.save(model,"ts_model/ts.pth")
                        print("better model have been saved!")
