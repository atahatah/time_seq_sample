import io

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pytorch_lightning as L
import wandb


class AirModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=50, num_layers=1, batch_first=True
        )
        self.linear = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


class AirModule(L.LightningModule):
    def __init__(
        self,
        lr=1e-4,
        lookback=4,
        save_interval=10,
        batches_to_log=1,
    ):
        super().__init__()
        self.lr = lr
        self.lookback = lookback
        self.save_interval = save_interval
        self.batches_to_log = batches_to_log

        self.model = AirModel()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, log_header="train", prog_bar=True)

    def validation_step(self, batch, batch_idx):
        save_result = (
            batch_idx < self.batches_to_log
            and self.current_epoch % self.save_interval == 0
        )
        self._step(
            batch, batch_idx, log_header="val", prog_bar=True, save_result=save_result
        )

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, log_header="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def _step(
        self,
        batch,
        batch_idx: int,
        log_header: str,
        save_result=False,
        prog_bar=False,
    ):
        x, y = batch
        y_hat: torch.Tensor = self.model(x)
        loss: torch.Tensor = self.criterion(y_hat, y)

        # log
        self.log(f"{log_header}_loss", loss.item())
        self.log(f"{log_header}_RMSE", loss.sqrt().item(), prog_bar=prog_bar)

        if not save_result:
            return loss

        N, L, F = x.size()

        # save result
        buf = io.BytesIO()

        ans_plot = y[:, -1, 0].detach().cpu().numpy()
        pred_plot = y_hat[:, -1, 0].detach().cpu().numpy()

        plt.plot(ans_plot, label="ans")
        plt.plot(pred_plot, label="pred")
        plt.legend()
        plt.savefig(buf, format="png")
        buf.seek(0)
        self.logger.experiment.log(
            {
                f"{log_header}_pred": wandb.Image(Image.open(buf)),
            }
        )
        plt.close()

        return loss
