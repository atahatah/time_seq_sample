import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import wandb

from time_seq.airline.src.module import AirModule
from time_seq.airline.src.data import AirDataModule


def main(opt: argparse.Namespace):
    L.seed_everything(opt.seed)

    module = AirModule(
        lr=opt.lr,
        lookback=opt.lookback,
        save_interval=opt.save_interval,
        batches_to_log=opt.batches_to_log,
    )
    dm = AirDataModule(
        batch_size=opt.batch_size,
        csv_path=opt.csv_file,
        row_name=opt.row_name,
        lookback=opt.lookback,
    )

    wandb_logger = WandbLogger(
        name=opt.name,
        project=opt.project,
        offline=opt.offline,
        save_dir="/work/time_seq/airline/wandb",
        log_model=not opt.offline,
        config=vars(opt),
        mode="disabled" if opt.offline else "online",
    )
    val_checkpoint = ModelCheckpoint(
        filename="best-{epoch}-{step}-{val_loss:.1f}",
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=opt.early_stopping_patience,
    )

    trainer = L.Trainer(
        max_epochs=opt.epoch,
        logger=wandb_logger,
        deterministic=True,
        callbacks=[val_checkpoint, early_stopping],
        limit_train_batches=opt.limit_batches,
        limit_val_batches=opt.limit_batches,
        limit_test_batches=opt.limit_batches,
        fast_dev_run=opt.fast_dev_run,
        val_check_interval=opt.val_check_interval,
        check_val_every_n_epoch=None,
    )
    trainer.fit(module, dm)
    trainer.test(module, dm, ckpt_path=val_checkpoint.best_model_path)

    filename = "/work/time_seq/airline/outputs/predict.png"
    show_prediction(
        module, dm, chpt_path=val_checkpoint.best_model_path, filename=filename
    )
    wandb_logger.experiment.log({"predict": wandb.Image(filename)})

    print("finishing...")
    wandb_logger.experiment.finish()
    print("done!")


def show_prediction(
    module: AirModule,
    dm: AirDataModule,
    chpt_path: str = None,
    filename: str = "predict.png",
):
    if chpt_path is not None:
        module = module.__class__.load_from_checkpoint(checkpoint_path=chpt_path)
    timeseries = dm.timeseries
    train_size = dm.train_size
    val_size = dm.val_size
    X_train = dm.train_set.tensors[0]
    X_val = dm.val_set.tensors[0]
    X_test = dm.test_set.tensors[0]
    lookback = dm.lookback
    with torch.no_grad():
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred = module(X_train)
        y_pred = y_pred[:, -1, :]
        train_plot[lookback:train_size] = module(X_train)[:, -1, :]

        # shift validation predictions for plotting
        val_plot = np.ones_like(timeseries) * np.nan
        val_plot[train_size + lookback : train_size + val_size] = module(X_val)[
            :, -1, :
        ]
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[train_size + val_size + lookback : len(timeseries)] = module(X_test)[
            :, -1, :
        ]
    # plot
    plt.plot(timeseries)
    plt.plot(train_plot, c="r")
    plt.plot(val_plot, c="g")
    plt.plot(test_plot, c="b")
    plt.savefig(filename)
    plt.close()


def create_parser():
    parser = argparse.ArgumentParser()
    # Learning parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=2000)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--limit_batches", type=float, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--csv_file", type=str, default="time_seq/airline/inputs/airline-passengers.csv"
    )
    parser.add_argument("--row_name", type=str, default="Passengers")

    parser.add_argument("--lookback", type=int, default=4)

    # Wandb parameters
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--project", type=str, default="time_seq")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--batches_to_log", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--val_check_interval", type=int, default=None)
    return parser


if __name__ == "__main__":
    opt = create_parser().parse_args()
    main(opt)
