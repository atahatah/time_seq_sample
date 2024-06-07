import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as L


def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i : i + lookback]
        target = dataset[i + 1 : i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


class SeqDataModule(L.LightningDataModule):
    def __init__(
        self,
        csv_path: str = "airline-passengers.csv",
        row_name: str = "Passengers",
        batch_size: int = 8,
        num_workers: int = 4,
        lookback: int = 4,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lookback = lookback
        self.row_name = row_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

    def setup(self, stage: str):
        df = pd.read_csv(self.csv_path)
        self.timeseries = df[[self.row_name]].values.astype("float32")

        # train-test split for time series
        self.train_size = int(len(self.timeseries) * self.train_ratio)
        self.val_size = int(len(self.timeseries) * self.val_ratio)
        self.test_size = len(self.timeseries) - self.train_size - self.val_size
        train, val, test = (
            self.timeseries[: self.train_size],
            self.timeseries[self.train_size : self.train_size + self.val_size],
            self.timeseries[self.train_size + self.val_size :],
        )

        X_train, y_train = create_dataset(train, lookback=self.lookback)
        X_val, y_val = create_dataset(val, lookback=self.lookback)
        X_test, y_test = create_dataset(test, lookback=self.lookback)
        self.train_set = data.TensorDataset(X_train, y_train)
        self.val_set = data.TensorDataset(X_val, y_val)
        self.test_set = data.TensorDataset(X_test, y_test)

    def train_dataloader(self):
        return data.DataLoader(
            self.train_set,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_set,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_set,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
