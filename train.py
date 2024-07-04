import argparse
import os
from pathlib import Path

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from models.base import MedicalNerModel
from utils.datasets import create_dataloader

class Train:
    """Class to handle training of the MedicalNerModel."""

    def __init__(self):
        """Initialize the Train class by parsing arguments and initializing the model."""
        self.args = self.parse_args()
        self.model = MedicalNerModel(self.args)

    def train(self):
        """Main method to handle the training process."""
        try:
            train_loader, val_loader = create_dataloader(self.args)
            self.args.train_loader = train_loader
            self.args.val_loader = val_loader

            # Limit the data size for debugging.
            limit_train_batches = None
            limit_val_batches = None
            if self.args.limit_batches > 0:
                limit_train_batches = self.args.limit_batches
                limit_val_batches = int(self.args.limit_batches * self.args.valid_ratio / (1 - self.args.valid_ratio))

            checkpoint_callback = ModelCheckpoint(
                dirpath=self.args.work_dir,
                filename="{epoch}-{val_acc:.4f}",
                save_top_k=5,
                monitor="val_acc",
                mode="max",
                save_last=True,
            )

            trainer = pl.Trainer(
                default_root_dir=self.args.work_dir,
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches,
                callbacks=[checkpoint_callback],
                max_epochs=self.args.epochs,
                min_epochs=self.args.epochs,
            )

            last_ckpt_path = self.args.work_dir / 'last.ckpt'
            if not self.args.resume or not os.path.exists(last_ckpt_path):
                last_ckpt_path = None

            trainer.fit(self.model,
                        train_dataloaders=train_loader,
                        val_dataloaders=val_loader,
                        ckpt_path=last_ckpt_path)
        except Exception as e:
            print(f"An error occurred during training: {e}")

    def parse_args(self):
        """Parse command-line arguments for training configuration."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=5e-5, help='Learning Rate.')
        parser.add_argument('--device', type=str, default='auto', help='The device for training. auto, cpu or cuda')
        parser.add_argument('--valid-ratio', type=float, default=0.1, help='The ratio of splitting validation set.')
        parser.add_argument('--batch-size', type=int, default=32, help='The batch size of training, validation, and test.')
        parser.add_argument('--workers', type=int, default=-1, help="The num_workers of dataloader. -1 means auto select.")
        parser.add_argument('--epochs', type=int, default=100, help='The number of training epochs.')
        parser.add_argument('--resume', action='store_true', help='Resume training.')
        parser.add_argument('--no-resume', dest='resume', action='store_false', help='Not resume training.')
        parser.set_defaults(resume=True)
        parser.add_argument('--limit-batches', type=int, default=-1, help='Limit the batches of datasets for quick testing if your model works. -1 means no limit.')
        parser.add_argument('--work-dir', type=str, default='./outputs', help='The path of output files while running, including model state file, tensorboard files, etc.')
        parser.add_argument('--num-labels', type=int, default=5, help='The number of labels for the classification task.')  # Add num_labels argument

        args = parser.parse_known_args()[0]

        if args.device == 'auto':
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            args.device = args.device

        print("Device:", args.device)

        if args.workers < 0:
            if args.device == 'cpu':
                args.workers = 0
            else:
                args.workers = os.cpu_count()

        args.work_dir = Path(args.work_dir)
        if not os.path.exists(args.work_dir):
            os.makedirs(args.work_dir)

        return args


if __name__ == '__main__':
    train = Train()
    train.train()
