from torchmetrics.functional import accuracy, precision, recall
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision.models import convnext_base, ConvNeXt_Base_Weights

def get_all_children(module, depth=0):
    if len(list(module.children())) == 0 or depth < 0:
        return [module]

    child_modules = list(module.children())
    subchild_modules = []
    for child_module in child_modules:
        subchild_modules += get_all_children(child_module, depth - 1)

    return subchild_modules

class PlantINaturalist2021FinetuneConvNeXtBase(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        for hyperparamater, value in config.items():
            setattr(self, hyperparamater, value)

        model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False

        if self.num_trainable_layers > 0:
            trainable_layers = get_all_children(model, depth=1)[:-4][-self.num_trainable_layers:]
        
            for layer in trainable_layers:
                for param in layer.parameters():
                    param.requires_grad = True


        in_features = model.classifier[2].in_features

        model.classifier[2] = nn.Linear(in_features, self.num_classes)

        for layer in model.classifier:
            for param in layer.parameters():
                param.requires_grad = True

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