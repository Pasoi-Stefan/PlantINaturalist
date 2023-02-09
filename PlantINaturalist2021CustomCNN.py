from torchmetrics.functional import accuracy, precision, recall
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn


class PlantINaturalist2021CustomCNN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        for hyperparamater, value in config.items():
            setattr(self, hyperparamater, value)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=2, stride=(2, 2)),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=2, stride=(2, 2)),
            nn.BatchNorm2d(num_features=12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=12, out_channels=48, kernel_size=2, padding='same'),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=2, padding='same'),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(48, self.num_classes)
        )


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.num_classes)
        prec = precision(y_hat, y, task="multiclass", average = 'macro', num_classes=self.num_classes)
        rec = recall(y_hat, y, task="multiclass", average = 'macro', num_classes=self.num_classes)
        metrics = {"train_acc": acc, "train_precision": prec, "train_recall": rec, "train_loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar = True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        acc = accuracy(y_hat, y, task="multiclass", num_classes=self.num_classes)
        prec = precision(y_hat, y, task="multiclass", average = 'macro', num_classes=self.num_classes)
        rec = recall(y_hat, y, task="multiclass", average = 'macro', num_classes=self.num_classes)
        metrics = {"val_acc": acc, "val_precision": prec, "val_recall": rec, "val_loss": loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar = True)
        return metrics

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.model.classifier.parameters(), lr=self.learning_rate, momentum=0.9)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        # return [optimizer], [lr_scheduler]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_epoch_step_size, gamma=self.lr_decay_rate)
        return [optimizer], [lr_scheduler]