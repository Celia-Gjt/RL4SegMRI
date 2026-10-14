import pickle
import random
from typing import Dict

import pytorch_lightning as pl
import torch
from torch import nn, optim

from rl4seg.utils.loss_functions import DC_and_BCE_loss
from vital.models.segmentation.unet import UNet

from rl4seg.utils.Metrics import accuracy, error_recall
from rl4seg.utils.logging_helper import log_image, log_case


class RewardOptimizer(pl.LightningModule):
    def __init__(self, save_model_path=None, uncertainty=False, var_file=None, **kwargs):
        super().__init__(**kwargs)

        self.net = UNet(input_shape=(2, 256, 256), output_shape=(1, 256, 256))

        self.uncertainty = uncertainty
        self.is_log_sigma = True
        self.iterations = 10

        self.temperature = nn.Parameter(torch.ones(1).to(self.device))
        self.var_file = var_file

        self.loss = nn.BCELoss()
        self.save_model_path = save_model_path

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val_loss"}

    def training_step(self, batch, *args, **kwargs) -> Dict:
        x, y = batch


        y_pred = torch.sigmoid(self(x))
        loss = self.loss(y_pred, y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs)
        return logs

    def validation_step(self, batch, batch_idx: int):
        x, y = batch

        y_pred = torch.sigmoid(self(x))
        loss = self.loss(y_pred, y)

        acc = accuracy(y_pred, x, y)
        rec = error_recall(y_pred, y)

        self.log_dict({"val_loss": loss,
                       "val_acc": acc.mean(),
                       'val_rec': rec.mean()})

        # log images
        idx = random.randint(0, len(x) - 1)  # which image to log
        im_show = x[idx, :1].permute((0, 2, 1)).detach().cpu().numpy()
        input_show = x[idx, 1:].permute((0, 2, 1)).detach().cpu().numpy()
        gt_show = y[idx].permute((0, 2, 1)).detach().cpu().numpy()
        pred_show = y_pred[idx].permute((0, 2, 1)).detach().cpu().numpy()
        log_case(self.logger, img=[im_show,input_show,gt_show,pred_show], title='Figure_reward', number=self.global_step,
                  img_text=rec[idx].mean())

        return {'loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = torch.sigmoid(self(x))
        loss = self.loss(y_pred, y)

        acc = accuracy(y_pred, x, y)
        rec = error_recall(y_pred, y)

        self.log_dict({"test_loss": loss,
                       "test_acc": acc.mean(),
                       'test_rec': rec.mean()})

        for i in range(len(x)):
            im_show = x[i, :1].permute((0, 2, 1)).detach().cpu().numpy()
            input_show = x[i, 1:].permute((0, 2, 1)).detach().cpu().numpy()
            gt_show = y[i].permute((0, 2, 1)).detach().cpu().numpy()
            pred_show = y_pred[i].permute((0, 2, 1)).detach().cpu().numpy()
            log_case(self.logger, img=[im_show,input_show,gt_show,pred_show], title='test_Figure_reward', number=batch_idx * (i + 1),
                      img_text=rec[i].mean())

        return {'loss': loss}

    def on_test_end(self) -> None:
        self.save_model()

    def save_model(self):
        if self.save_model_path:
            sd = self.net.state_dict()
            torch.save(sd, self.save_model_path)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        #temperature = self.temperature.unsqueeze(1).expand(logits.size()).to(self.device)
        return torch.div(logits, self.temperature)

    def on_train_end(self) -> None:
        val_loader = self.trainer.datamodule.val_dataloader()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in val_loader:
                logits = self(input.to(self.device))
                logits_list.append(torch.stack((1-logits, logits), dim=1).squeeze(2))
                labels_list.append(label.squeeze(1).to(torch.long))
            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device)

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=10000)
        criterion = nn.CrossEntropyLoss().to(self.device)

        def eval():
            optimizer.zero_grad()
            loss = criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        print(f"TEMPERATURE: {self.temperature}")
        self.trainer.logger.log_hyperparams({'Temperature factor': self.temperature.detach().cpu().numpy()[0]})

        if self.var_file:
            pickle.dump({"Temperature_factor": self.temperature.detach().cpu().numpy()[0]}, open(self.var_file, "wb"))

