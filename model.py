import torch
import torch.nn as nn
import lightgbm as lgb
import numpy as np
from torchensemble import VotingRegressor
from torchensemble._base import torchensemble_model_doc
from torchensemble.utils import set_module, io
from torchensemble.utils import operator as op
from joblib import Parallel, delayed
import warnings
#from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
#from models.s4.s4d import S4D

def _parallel_fit_per_epoch(
    train_loader,
    estimator,
    cur_lr,
    optimizer,
    criterion,
    idx,
    epoch,
    log_interval,
    device,
    is_classification,
):
    """
    Private function used to fit base estimators in parallel.

    WARNING: Parallelization when fitting large base estimators may cause
    out-of-memory error.
    """
    if cur_lr:
        # Parallelization corrupts the binding between optimizer and scheduler
        set_module.update_lr(optimizer, cur_lr)

    for batch_idx, elem in enumerate(train_loader):

        data, target = io.split_data_target(elem, device)
        batch_size = data[0].size(0)

        optimizer.zero_grad()
        output = estimator(*data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Print training status
        if batch_idx % log_interval == 0:

            # Classification
            if is_classification:
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == target).sum().item()

                msg = (
                    "Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                    " | Loss: {:.5f} | Correct: {:d}/{:d}"
                )
                print(
                    msg.format(
                        idx, epoch, batch_idx, loss, correct, batch_size
                    )
                )
            # Regression
            else:
                msg = (
                    "Estimator: {:03d} | Epoch: {:03d} | Batch: {:03d}"
                    " | Loss: {:.5f}"
                )
                print(msg.format(idx, epoch, batch_idx, loss))

    return estimator, optimizer, loss

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
        return out * self.model_weight


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
        self.model_weight = nn.Parameter(torch.tensor(1.))
    
    def forward(self, x):
        out, _ = self.lstm(x)
        predictions = self.fc(out).squeeze()
        return predictions * self.model_weight

   
class transMODEL(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(transMODEL, self).__init__()
        self.hidden_size = hidden_size
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(input_size, 1)
        self.model_weight = nn.Parameter(torch.tensor(1.))
    
    def forward(self, x):
        # x = x.unsqueeze(1)

        out = self.transformer_encoder(x)
        out = self.fc(out)  # 取最后一个时间步的输出作为预测
        return out * self.model_weight

class mlp(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dr=0.0):
        super(mlp, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.drop = nn.Dropout(dr)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.model_weight = nn.Parameter(torch.tensor(1.))
 
    def forward(self, input):
        return self.linear2(self.relu(self.drop(self.linear1(input)))) * self.model_weight
    



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
        self.model_weight = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        lgb_input = x.cpu().numpy()
        h,y,z=lgb_input.shape
        lgb_input = np.reshape(lgb_input,(-1,z))
        pred = self.lgb_model.predict(lgb_input)
        pred = np.reshape(np.squeeze(pred),(h,y))
        return torch.from_numpy(pred).to(x.device) * self.model_weight

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

def loading_bagging(file_name):
    model= torch.load(file_name)
    return model

def loading_voting(file_name):
    model= torch.load(file_name)
    return model

LOADING_FUNC_MAPPING = {
    'lightgbm': loading_lightgmb,
    'transformer': loading_transformer,
    'lstm': loading_lstm
}

def enable_model_weight(model: nn.Module):
    for pn, p in model.named_parameters():
        print(pn, p.requires_grad)

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

def get_models(ensamble_models, ensamble_ckpt, args, device):
    models = {mm: LOADING_FUNC_MAPPING[mm](mp, args) for mm, mp in zip(ensamble_models, ensamble_ckpt)}
    models = {mm: models[mm].to(device) if hasattr(models[mm], 'to') else models[mm] for mm in models}
    models = [models[mm].requires_grad_(False) if hasattr(models[mm], 'requires_grad_') else models[mm] for mm in models] # frozen models
    models = [enable_model_weight(mm) for mm in models]

class VotingEnsamble(VotingRegressor):

    def setting_models(self, estimate_models, loss_fn):
        self.estimate_models = estimate_models
        self.loss_fn = loss_fn

    @torchensemble_model_doc(
        """Implementation on the training stage of VotingRegressor.""", "fit"
    )
    def fit(
        self,
        train_loader,
        epochs=100,
        log_interval=100,
        test_loader=None,
        save_model=True,
        save_dir=None,
    ):

        self._validate_parameters(epochs, log_interval)
        self.n_outputs = self._decide_n_outputs(train_loader)

        # Instantiate a pool of base estimators, optimizers, and schedulers.
        estimators = self.estimate_models
        # for _ in range(self.n_estimators):
        #     estimators.append(self._make_estimator())
        self.n_estimators = len(estimators)

        optimizers = []
        for i in range(self.n_estimators):
            optimizers.append(
                set_module.set_optimizer(
                    estimators[i], self.optimizer_name, **self.optimizer_args
                )
            )

        if self.use_scheduler_:
            scheduler_ = set_module.set_scheduler(
                optimizers[0], self.scheduler_name, **self.scheduler_args
            )

        # Check the training criterion
        # if not hasattr(self, "_criterion"):
        #     self._criterion = nn.MSELoss()
        self._criterion = self.loss_fn

        # Utils
        best_loss = float("inf")

        # Internal helper function on pseudo forward
        def _forward(estimators, *x):
            outputs = [estimator(*x) for estimator in estimators]
            pred = op.average(outputs)

            return pred

        # Maintain a pool of workers
        with Parallel(n_jobs=self.n_jobs) as parallel:

            # Training loop
            for epoch in range(epochs):
                self.train()

                if self.use_scheduler_:
                    if self.scheduler_name == "ReduceLROnPlateau":
                        cur_lr = optimizers[0].param_groups[0]["lr"]
                    else:
                        cur_lr = scheduler_.get_last_lr()[0]
                else:
                    cur_lr = None

                if self.n_jobs and self.n_jobs > 1:
                    msg = "Parallelization on the training epoch: {:03d}"
                    self.logger.info(msg.format(epoch))

                rets = parallel(
                    delayed(_parallel_fit_per_epoch)(
                        train_loader,
                        estimator,
                        cur_lr,
                        optimizer,
                        self._criterion,
                        idx,
                        epoch,
                        log_interval,
                        self.device,
                        False,
                    )
                    for idx, (estimator, optimizer) in enumerate(
                        zip(estimators, optimizers)
                    )
                )

                estimators, optimizers, losses = [], [], []
                for estimator, optimizer, loss in rets:
                    estimators.append(estimator)
                    optimizers.append(optimizer)
                    losses.append(loss)

                # Validation
                if test_loader:
                    self.eval()
                    with torch.no_grad():
                        val_loss = 0.0
                        for _, elem in enumerate(test_loader):
                            data, target = io.split_data_target(
                                elem, self.device
                            )
                            output = _forward(estimators, *data)
                            val_loss += self._criterion(output, target)
                        val_loss /= len(test_loader)

                        if val_loss < best_loss:
                            best_loss = val_loss
                            self.estimators_ = nn.ModuleList()
                            self.estimators_.extend(estimators)
                            if save_model:
                                io.save(self, save_dir, self.logger)

                        msg = (
                            "Epoch: {:03d} | Validation Loss:"
                            " {:.5f} | Historical Best: {:.5f}"
                        )
                        self.logger.info(
                            msg.format(epoch, val_loss, best_loss)
                        )
                        if self.tb_logger:
                            self.tb_logger.add_scalar(
                                "voting/Validation_Loss", val_loss, epoch
                            )
                # No validation
                else:
                    self.estimators_ = nn.ModuleList()
                    self.estimators_.extend(estimators)
                    if save_model:
                        io.save(self, save_dir, self.logger)

                # Update the scheduler
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)

                    if self.use_scheduler_:
                        if self.scheduler_name == "ReduceLROnPlateau":
                            if test_loader:
                                scheduler_.step(val_loss)
                            else:
                                loss = torch.mean(torch.tensor(losses))
                                scheduler_.step(loss)
                        else:
                            scheduler_.step()
