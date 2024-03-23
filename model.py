import torch
import torch.nn as nn
import lightgbm as lgb
import numpy as np
#from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
#from models.s4.s4d import S4D

class rnnMODEL(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(rnnMODEL, self).__init__()
        self.hidden_size = hidden_size
        # self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size,1)
        self.model_weight = nn.Parameter(torch.tensor(1.))
    
    def forward(self, x):
        # x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.fc(out)  # 取最后一个时间步的输出作为预测
        return out


# class lstmMODEL(nn.Module):
#     def __init__(self,input_size,hidden_size):
#         super(lstmMODEL, self).__init__()
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         # self.lstm = nn.RNN(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size,1)
    
#     def forward(self, x):
#         # x = x.unsqueeze(1)
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出作为预测
#         return out

class lstmMODEL(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstmMODEL, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.model_weight
    
    def forward(self, x):
        out, _ = self.lstm(x)
        predictions = self.fc(out).squeeze()
        return predictions

   
class transMODEL(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(transMODEL, self).__init__()
        self.hidden_size = hidden_size
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, x):
        # x = x.unsqueeze(1)

        out = self.transformer_encoder(x)
        out = self.fc(out)  # 取最后一个时间步的输出作为预测
        return out

class mlp(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dr=0.0):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.drop = nn.Dropout(dr)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
 
    def forward(self, input):
        return self.linear2(self.relu(self.drop(self.linear1(input))))
    



if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, 1e-5))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

class lgbModule(nn.Module):
    def __init__(self, lgb_model) -> None:
        super().__init__()
        self.lgb_model = lgb_model
        self.model_weight

    def forward(self, x):
        lgb_input = x.cpu().numpy()
        h,y,z=lgb_input.shape
        lgb_input = np.reshape(lgb_input,(-1,z))
        pred = self.lgb_model.predict(lgb_input)
        pred = np.reshape(np.squeeze(pred),(h,y))
        return torch.from_numpy(pred).to(x.device)

def loading_lightgmb(file_name, args):
    return lgbModule(lgb.Booster(model_file=file_name))

def loading_transformer(file_name, args):
    model = transMODEL(args.nvars,args.hidden_size)
    model.load_state_dict(torch.load(file_name))
    return model

def loading_lstm(file_name, args):
    model = lstmMODEL(args.nvars,args.hidden_size)
    model.load_state_dict(torch.load(file_name), strict = False)
    return model

LOADING_FUNC_MAPPING = {
    'lightgbm': loading_lightgmb,
    'transformer': loading_transformer,
    'lstm': loading_lstm
}

def enable_model_weight(model):
    for mn, mm in model.named_modules():
        if hasattr(mm, 'model_weight'):
            mm.model_weight.requires_grad_(True)
    return model

class EnsambleModel(nn.Module):
    def __init__(self, ensamble_models, ensamble_ckpt, args, device, 
                 ensamble_style, manual_weight) -> None:
        super().__init__()
        self.models = {mm: LOADING_FUNC_MAPPING[mm](mp, args) for mm, mp in zip(ensamble_models, ensamble_ckpt)}
        self.models = {mm: self.models[mm].to(device) if hasattr(self.models[mm], 'to') else self.models[mm] for mm in self.models}
        self.ensamble_style = ensamble_style
        if ensamble_style == 'manual_weight':
            manual_weight = np.mean(manual_weight).tolist()
            self.manual_weight = {mm: mw for mm, mw in zip(ensamble_models, manual_weight)}
        # self.requires_grad_(False)
        self.models = [self.models[mm].requires_grad_(False) if hasattr(self.models[mm], 'requires_grad_') else self.models[mm] for mm in self.models] # frozen models
        self.models = [enable_model_weight(mm) for mm in self.models]

    def forward(self, x):
        # default_weight = 1/len(self.models)
        ret = None
        if self.ensamble_style == 'manual_weight':
            for model_name in self.models:
                if ret == None:
                    ret = self.models[model_name](x) * self.manual_weight[model_name]
                else:
                    ret += self.models[model_name](x) * self.manual_weight[model_name]
            return ret
