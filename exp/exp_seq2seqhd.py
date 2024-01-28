import warnings
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from exp.exp_basic import Exp_Basic
from models import MultivariateSeq2SeqModel
from utils.metrics import cumavg, metric

warnings.filterwarnings("ignore")


class net(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.model = MultivariateSeq2SeqModel(
            T=args.seq_len, D=args.hvs_len, tau=args.pred_len
        ).to(self.device)

    def forward(self, x):
        return self.model(x)


class ExpSeq2SeqHD(Exp_Basic):
    def __init__(self, args):
        super(ExpSeq2SeqHD, self).__init__(args)
        self.args = args
        self.model = net(args, device=self.device)

    def _get_data(self, flag):
        args = self.args

        data_dict_ = {
            "ETTh1": Dataset_ETT_hour,
            "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
            "WTH": Dataset_Custom,
            "ECL": Dataset_Custom,
            "ILI": Dataset_Custom,
            "S-A": Dataset_Custom,
            "custom": Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 2

        freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=False,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
        )
        print(flag, len(data_set))

        return data_set

    def _select_optimizer(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _select_criterion(self):
        return nn.HuberLoss()

    def train(self):
        tau, Ts = self.args.pred_len, self.args.seq_len

        train_data: np.ndarray = self._get_data(flag="train").data_x
        self._select_optimizer()
        for i in tqdm(range(Ts, train_data.shape[0] - tau, 1)):
            self._process_one_batch(train_data, i, mode="train")

    def test(self):
        test_data: np.ndarray = self._get_data(flag="test").data_x

        preds = []
        trues = []
        rses, corrs = [], []
        for i in tqdm(
            range(
                self.args.seq_len,
                test_data.shape[0] - self.args.pred_len,
                self.args.pred_len,
            )
        ):
            pred, true = self._process_one_batch(test_data, i, mode="test")
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            rse, corr = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            rses.append(rse)
            corrs.append(corr)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print("test shape:", preds.shape, trues.shape)

        RSE, CORR = cumavg(rses), cumavg(corrs)
        rse, corr = RSE[-1], CORR[-1]

        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        print(f"rse:{rse}, corr:{corr}")
        return [rse, corr], RSE, CORR, preds, trues

    def _process_one_batch(
        self, data: np.ndarray, idx: int, mode: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        x_seq = torch.Tensor(data[idx - self.args.seq_len : idx, :]).to(self.device)
        y = torch.Tensor(data[idx : idx + self.args.pred_len, :]).to(self.device)
        self.opt.zero_grad()
        seq_tilda = self.model(x_seq.T).T

        loss = self._select_criterion()(seq_tilda, y)
        l2_reg = torch.tensor(0.0).to(self.device)
        for param in self.model.parameters():
            l2_reg += torch.norm(param) ** 2
        # Add L2 regularization to the loss
        loss += self.args.l2_lambda * l2_reg
        loss.backward()
        self.opt.step()
        if mode == "test":
            return seq_tilda, y
        else:
            return None
