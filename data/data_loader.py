import os
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.timefeatures import time_features
from utils.tools import StandardScaler

warnings.filterwarnings("ignore")


class Dataset_TSFHD(Dataset):
    def __init__(
        self,
        root_path: str,
        flag: str = "train",
        size: Optional[List[int]] = None,
        features: str = "S",
        data_path: str = "ETTh1.csv",
        target: str = "OT",
        scale: bool = True,
        inverse: bool = False,
        timeenc: int = 0,
        freq: str = "h",
        cols: Optional[List[str]] = None,
    ) -> None:
        """
        Dataset for Time Series Forecasting with Historical Data.

        Args:
            root_path (str): Root path for the dataset.
            flag (str): Flag indicating the dataset split ("train", "test", "val").
            size (List[int], optional): Size of sequence, label, and prediction length.
            features (str): Type of features ("S", "M", or "MS").
            data_path (str): Path to the dataset file.
            target (str): Target feature name.
            scale (bool): Whether to scale the data.
            inverse (bool): Whether to inverse transform the data.
            timeenc (int): Type of time encoding.
            freq (str): Frequency of the time series data.
            cols (List[str], optional): List of column names.
        """
        # Initialize default values for sequence length, label length, and prediction length
        self.seq_len = 24 * 4 * 4
        self.label_len = 24 * 4
        self.pred_len = 24 * 4

        # Update values if provided
        if size is not None:
            self.seq_len, self.label_len, self.pred_len = size

        # Initialize class attributes
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self) -> None:
        """
        Read and preprocess the dataset.
        """
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Select relevant columns
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]

        # Split dataset based on flag
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select features based on feature type
        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        # Scale the data if needed
        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract timestamp features
        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        # Set class attributes
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get items for a given index.

        Args:
            index (int): Index to retrieve items.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Sequence x, sequence y, sequence x mark, sequence y mark.
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [
                    self.data_x[r_begin : r_begin + self.label_len],
                    self.data_y[r_begin + self.label_len : r_end],
                ],
                0,
            )
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform the scaled data.

        Args:
            data (np.ndarray): Scaled data.

        Returns:
            np.ndarray: Inverse transformed data.
        """
        return self.scaler.inverse_transform(data)
