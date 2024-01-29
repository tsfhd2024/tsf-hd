# TSF-HD

[![Python](https://img.shields.io/badge/Python-3.10.12-blue.svg)](https://www.python.org/downloads/release/python-31012/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.1.2-blue.svg)](https://pytorch.org/)



This project contains the Pytorch implementation of the following paper:

**Title:** A Novel Hyperdimensional Computing Framework for  Online Time Series Forecasting on the Edge

**Authors:** [Mohamed Mejri](mohamed.mejri@gatech.edu), [Chandramouli Amarnath](chandamarnath@gatech.edu) and [Abhijit Chatterjee](abhijit.chatterjee@ece.gatech.edu)

## Introduction

We present a novel framework for efficient time series forecasting in edge computing. It departs from traditional, resource-intensive deep learning methods, embracing hyperdimensional computing (HDC) for a more efficient approach. The framework includes two models: the Autoregressive Hyperdimensional Computing (AR-HDC) and the Sequence-to-Sequence HDC (Seq2Seq-HDC). These models are designed to reduce inference times and improve accuracy in both short-term and long-term forecasting, making them ideal for resource-limited edge computing scenarios

![TSF-HD](https://github.com/tsfhd2024/tsf-hd/blob/main/data/image/AR-Seq2Seq-Overview.png)

## Requirements

* Python 3.10.12
* numpy==1.26.3
* pandas==2.1.4
* scikit-learn==1.3.2
* torch==2.1.2
* tqdm==4.66.1

Please install the required packages listed in the requirements.txt file using the following command :

``` bash
pip install -r requirements.txt
```
## Benchmarking

### 1. Data preparation

We follow the same data formatting as the Informer repo (https://github.com/zhouhaoyi/Informer2020), which also hosts the raw data.
Please put all raw data (csv) files in the ```./data``` folder.

### 2. Run experiments

To replicate our results on the ETT, ECL, Exchange, Illness, and WTH datasets, run
```
chmod +x scripts/*.sh
bash .scripts/run.sh
```

### 3.  Arguments

**Method:** Our implementation supports the following training strategies:
- AR-HDC: Autoregressive Hyperdimensional Computing Framework
- Seq2Seq-HDC: Sequence-to-Sequence Hyperdimensional Computing Framework


You can specify one of the above method via the ```--method``` argument.

**Dataset:** Our implementation currently supports the following datasets: Electricity Transformer - ETT (including ETTh1, ETTh2, ETTm1, and ETTm2), ECL, Exchange, Illness and WTH. You can specify the dataset via the ```--data``` argument.

**Other arguments:** Other useful arguments for experiments are:
- ```--hvs_len```: Dimension of the hyperspace: e.g. **D = 1000** ,
- ```--seq_len```: look-back windows' length, set to **2 * &tau;** by default,
- ```--pred_len```: forecast windows' **&tau** length

 ### 4.  Results
![TSF-HD](https://github.com/tsfhd2024/tsf-hd/blob/main/data/image/short_term.png)
![TSF-HD](https://github.com/tsfhd2024/tsf-hd/blob/main/data/image/long_term.png)
