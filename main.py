import argparse
import os

import numpy as np
import torch

from exp import ExpARHD, ExpSeq2SeqHD


def main():
    parser = argparse.ArgumentParser(
        description="[Informer] Long Sequences Forecasting"
    )

    parser.add_argument("--data", type=str, required=True, default="ETTh1", help="data")
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")

    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=96,
        help="input sequence length of Informer encoder",
    )
    parser.add_argument(
        "--pred_len", type=int, default=24, help="prediction sequence length"
    )
    parser.add_argument(
        "--hvs_len", type=int, default=24, help="dimension of the hypervectors"
    )
    parser.add_argument(
        "--label_len",
        type=int,
        default=48,
        help="start token length of Informer decoder",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    # parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument(
        "--cols",
        type=str,
        nargs="+",
        help="certain cols from the data files as the input features",
    )

    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

    parser.add_argument("--test_bsz", type=int, default=-1)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    )
    parser.add_argument(
        "--method", type=str, default="seq2seq-HDC", help="choose seq2seq-HDC or AR-HDC"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="choose learning rate : default 1e-3",
    )
    parser.add_argument(
        "--l2_lambda",
        type=float,
        default=2e-3,
        help="choose regularization rate l2 parameter: default 2e-3",
    )

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.test_bsz = args.batch_size if args.test_bsz == -1 else args.test_bsz
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        "ETTh1": {
            "data": "ETTh1.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "ETTh2": {
            "data": "ETTh2.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "ETTm1": {
            "data": "ETTm1.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "ETTm2": {
            "data": "ETTm2.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "WTH": {
            "data": "WTH.csv",
            "T": "WetBulbCelsius",
            "M": [12, 12, 12],
            "S": [1, 1, 1],
            "MS": [12, 12, 1],
        },
        "ECL": {
            "data": "ECL.csv",
            "T": "MT_320",
            "M": [321, 321, 321],
            "S": [1, 1, 1],
            "MS": [321, 321, 1],
        },
        "S-A": {"data": "Toy.csv", "T": "Value", "S": [1, 1, 1]},
        "ToyG": {"data": "ToyG.csv", "T": "Value", "S": [1, 1, 1]},
        "Exchange": {"data": "exchange_rate.csv", "T": "TND", "M": [8, 8, 8]},
        "Illness": {"data": "national_illness.csv", "T": "OT", "M": [7, 7, 7]},
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info["data"]
        args.target = data_info["T"]

        # Exp = Exp_TS2VecSupervised
    Exps = {"seq2seq-HDC": ExpSeq2SeqHD, "AR-HDC": ExpARHD}
    Exp = Exps[args.method]
    metrics, preds, true, corr, rse = [], [], [], [], []

    low = 0
    high = 10
    seeds = torch.randint(low, high + 1, size=(args.itr,))
    for ii in range(args.itr):
        torch.manual_seed(seeds[ii].item())

        # Set the seed for GPU if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds[ii].item())

        exp = Exp(args)  # set experiments
        exp.train()

        m, rse_, corr_, p, t = exp.test()
        metrics.append(m)
        preds.append(p)
        true.append(t)

        corr.append(corr_)
        rse.append(rse_)
        torch.cuda.empty_cache()

        # folder_path = './results/' + setting + '/'
        folder_path = "./results_{}/{}/".format(args.method, args.itr)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    np.save(folder_path + "metrics.npy", np.array(metrics))
    np.save(folder_path + "preds.npy", np.array(preds))
    np.save(folder_path + "trues.npy", np.array(true))
    np.save(folder_path + "corr.npy", np.array(corr))
    np.save(folder_path + "rse.npy", np.array(rse))


if __name__ == "__main__":
    main()
