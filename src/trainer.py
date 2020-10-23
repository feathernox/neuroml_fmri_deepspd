import geoopt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ConnectivityNetworksClassifier(pl.LightningModule):
    def __init__(self, net, train_dataset, val_dataset, opt='adam', opt_lr=0.1, opt_kwargs=None,
                 batch_size=32, num_workers=0):
        super(ConnectivityNetworksClassifier, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.net = net
        self.opt = opt
        self.opt_lr = opt_lr
        self.opt_kwargs = opt_kwargs

        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, x):
        logits = self.net(x)
        return logits

    def train_dataloader(self):
        dataloader_train = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers, shuffle=True)
        return dataloader_train

    def val_dataloader(self):
        dataloader_val = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                    num_workers=self.num_workers, shuffle=False)
        return dataloader_val

    def configure_optimizers(self):
        if self.opt == 'sgd':
            optimizer = geoopt.optim.RiemannianSGD(self.parameters(), self.opt_lr, **self.opt_kwargs)
        elif self.opt == 'adam':
            optimizer = geoopt.optim.RiemannianAdam(self.parameters(), self.opt_lr, **self.opt_kwargs)
        else:
            raise ValueError('Optimizer not supported')

        return optimizer

    def training_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.forward(data)
        labels = labels.unsqueeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels.double())
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        data, labels = val_batch
        logits = self.forward(data)
        labels = labels.unsqueeze(-1)
        pred = (torch.sigmoid(logits) > 0.5).long()
        loss = F.binary_cross_entropy_with_logits(logits, labels.double(), reduction='sum')
        correct = (pred == labels).sum()
        return {'val_loss': loss, 'val_correct': correct}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum() / len(self.val_dataset)
        acc = torch.stack([x['val_correct'] for x in outputs]).sum().float() / len(self.val_dataset)
        logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'loss': avg_loss, 'log': logs}
