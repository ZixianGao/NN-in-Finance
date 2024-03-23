import torch
import torch.nn as nn
import lightgbm as lgb
import numpy as np


class rnnMODEL(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(rnnMODEL, self).__init__()
        self.hidden_size = hidden_size
        # self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
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

    def forward(self, x):
        out, _ = self.lstm(x)
        #out = out[:, -1, :]
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


if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class lgbModule(nn.Module):
    def __init__(self, lgb_model) -> None:
        super().__init__()
        self.lgb_model = lgb_model

    def forward(self, x):
        lgb_input = x.cpu().numpy()
        h, y, z = lgb_input.shape
        lgb_input = np.reshape(lgb_input, (-1, z))
        pred = self.lgb_model.predict(lgb_input)
        pred = np.reshape(np.squeeze(pred), (h, y))
        return torch.from_numpy(pred).to(x.device)


def loading_lightgbm(file_name, args):
    return lgbModule(lgb.Booster(model_file=file_name))


def loading_transformer(file_name, args):
    model = transMODEL(args.nvars, args.hidden_size)
    model.load_state_dict(torch.load(file_name))
    return model


def loading_lstm(file_name, args):
    model = lstmMODEL(args.nvars, args.hidden_size)
    model.load_state_dict(torch.load(file_name), strict=False)
    return model


LOADING_FUNC_MAPPING = {
    'lightgbm': loading_lightgbm,
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
        self.models = {mm: self.models[mm].to(device) if hasattr(self.models[mm], 'to') else self.models[mm] for mm in
                       self.models}
        self.ensamble_style = ensamble_style
        if ensamble_style == 'manual_weight':
            manual_weight = np.mean(manual_weight).tolist()
            self.manual_weight = {mm: mw for mm, mw in zip(ensamble_models, manual_weight)}
        # self.requires_grad_(False)
        self.models = [
            self.models[mm].requires_grad_(False) if hasattr(self.models[mm], 'requires_grad_') else self.models[mm] for
            mm in self.models]  # frozen models
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
