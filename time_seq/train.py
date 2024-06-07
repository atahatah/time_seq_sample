from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib

import lightning as L

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm = nn.LSTM(1, 51, 2, batch_first=True)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        N, L, F = input.size()
        outputs, (h_n, c_n) = self.lstm(input)
        outputs = self.linear(outputs)

        future_outputs = torch.zeros(N, L + future, F)
        future_outputs[:, :L, :] = outputs
        future_intpus = torch.zeros_like(future_outputs)
        future_intpus[:, :L, :] = input
        for i in range(future):  # if we should predict the future
            future_output, (h, c) = self.lstm(future_intpus[:, : L + i], (h_n, c_n))
            future_output = self.linear(future_output)
            future_outputs += [future_output]
        future_outputs = torch.cat(future_outputs, dim=1)
        outputs = torch.cat([outputs, future_outputs], dim=1)
        return outputs


class SequenceModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Sequence()
        self.criterion = nn.MSELoss()
        self.lr = 1e-4
        self.future = 1000

    def training_step(self, batch, batch_idx):
        input, target = batch
        input = input.transpose(1, 0).unsqueeze(-1)
        out = self.model(input)
        loss = self.criterion(out, target)
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss
        # return self._step(
        #     batch,
        #     batch_idx,
        #     log_header="train",
        # )

    def validation_step(self, batch, batch_idx):
        test_input, test_target = batch
        test_input = test_input.transpose(1, 0).unsqueeze(-1)
        pred = self.model(test_input, future=self.future)
        loss = self.criterion(pred[:, : -self.future], test_target)
        self.log("val_loss", loss.item(), prog_bar=True)
        y = pred.detach().numpy()

        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title(
            "Predict future values for time sequences\n(Dashlines are predicted values)",
            fontsize=30,
        )
        plt.xlabel("x", fontsize=20)
        plt.ylabel("y", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            plt.plot(
                np.arange(input.size(1)), yi[: input.size(1)], color, linewidth=2.0
            )
            plt.plot(
                np.arange(input.size(1), input.size(1) + self.future),
                yi[input.size(1) :],
                color + ":",
                linewidth=2.0,
            )

        draw(y[0], "r")
        draw(y[1], "g")
        draw(y[2], "b")
        plt.savefig("predict%d.pdf" % batch_idx)
        plt.close()

        # save_samples = (
        #     batch_idx < self.batches_to_log
        #     and self.current_epoch % self.save_interval == 0
        # )
        # self._step(
        #     batch,
        #     batch_idx,
        #     log_header="val",
        #     save_samples=save_samples,
        # )

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, log_header="test")

    def configure_optimizers(self):
        return torch.optim.Adam(lr=self.lr, params=self.model.parameters())

    def forward(self, x):
        return self.model(x)

    def _step(
        self,
        batch,
        batch_idx: int,
        log_header: str,
        save_samples=False,
    ):
        # x, y = batch
        # pred = self.model(x)
        # loss = self.criterion(pred[:, :-future], y)
        # self.log(f"{log_header}_loss", loss)
        # return loss
        pass


def main():
    L.seed_everything(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15, help="steps to run")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size for training"
    )
    opt = parser.parse_args()

    # load data and make training set
    data = torch.load("time_seq/traindata.pt")
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(input, target), batch_size=opt.batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_input, test_target),
        batch_size=opt.batch_size,
    )

    trainer = L.Trainer(max_epochs=opt.steps)
    model = SequenceModel()
    trainer.fit(model, train_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
