from torchmetrics.functional import accuracy, precision, recall
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from torchvision.models.mobilenet import mobilenet_v2, MobileNet_V2_Weights

TRANSFORM = transforms.Compose([
    # transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TRANSFORM2 = transforms.Compose([
    transforms.AutoAugment(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class PlantINaturalist2021FinetuneMobileNetv2(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        for hyperparamater, value in config.items():
            setattr(self, hyperparamater, value)

        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        for param in model.parameters():
            param.requires_grad = False

        trainable_layers = list(model.features.children())[-self.num_trainable_layers:]
    
        for layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True

        # by default trainable
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(model.last_channel, self.num_classes),
        )

        self.model = model


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