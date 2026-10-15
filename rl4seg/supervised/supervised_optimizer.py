import copy
import random
from datetime import datetime
from typing import Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, Tensor

from rl4seg.utils.loss_functions import DC_and_CE_loss
from vital.models.segmentation.unet import UNet
from vital.data.pepr.config import Label

from rl4seg.utils.Metrics import accuracy, dice_score, is_anatomically_valid_mri
from rl4seg.utils.file_utils import save_to_reward_dataset
from rl4seg.utils.logging_helper import log_case
from rl4seg.utils.tensor_utils import convert_to_numpy
from rl4seg.utils.test_metrics import dice, hausdorff


class SupervisedOptimizer(pl.LightningModule):
    def __init__(self, input_shape=(1, 256, 256), output_shape=(1, 256, 256), loss=DC_and_CE_loss(soft_dice_kwargs={"batch_dice": True, "smooth": 1e-5, "do_bg": False}, ce_kwargs={}, weight_ce=1, weight_dice=1), ckpt_path=None, corrector=None, predict_save_dir=None, **kwargs):
        super().__init__(**kwargs)

        self.net = UNet(input_shape=input_shape, output_shape=output_shape)
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.loss = loss
        self.save_test_results = False
        self.ckpt_path = ckpt_path
        self.predict_save_dir = predict_save_dir
        self.pred_corrector = corrector

    def forward(self, x):
        out = self.net.forward(x)
        if self.output_shape[0] > 1:
            out = torch.softmax(out, dim=1)
        else:
            out = torch.sigmoid(out).squeeze(1)
        return out

    def configure_optimizers(self):
        # add weight decay so predictions are less certain, more randomness?
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0)

    def training_step(self, batch: dict[str, Tensor], *args, **kwargs) -> Dict:
        x, y = batch['img'], batch['gt']

        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        logs = {
            'loss': loss,
        }

        self.log_dict(logs, batch_size=x.shape[0], prog_bar=True)
        return logs

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        b_img, b_gt = batch['img'], batch['gt']
        y_pred = self.forward(b_img)

        loss = self.loss(y_pred, b_gt)

        if self.output_shape[0] > 1:
            y_pred = y_pred.argmax(dim=1)
        else:
            y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt)
        dice = dice_score(y_pred, b_gt, self.output_shape[0])
        #anat_err = has_anatomical_error(y_pred)

        logs = {'val_loss': loss,
                'val_acc': acc.mean(),
                'val_dice': dice.mean(),
                #'val_anat_errors': anat_err.mean(),
                }

        # log images
        idx = random.randint(0, len(b_img) - 1)  # which image to log

        im_show = b_img[idx].permute((0, 2, 1)).detach().cpu().numpy()
        gt_show = b_gt[idx].unsqueeze(0).permute((0, 2, 1)).detach().cpu().numpy()
        pred_show = y_pred[idx].unsqueeze(0).permute((0, 2, 1)).detach().cpu().numpy()
        log_case(self.logger, img=[im_show, gt_show, pred_show], title='Figure_supervised',
                 number=batch_idx * (idx + 1),
                 img_text=dice[idx].mean())

        self.log_dict(logs, batch_size=b_img.shape[0])
        return logs

    def test_step(self, batch, batch_idx):
        b_img, b_gt, voxel_spacing, is_base = batch['img'], batch['gt'], batch['vox'], batch['is_base']
        y_pred = self.forward(b_img)
        loss = self.loss(y_pred, b_gt)

        if self.output_shape[0] > 1:
            y_pred = y_pred.argmax(dim=1)
        else:
            y_pred = torch.round(y_pred)

        acc = accuracy(y_pred, b_img, b_gt)
        simple_dice = dice_score(y_pred, b_gt, self.output_shape[0])
        y_pred_np = y_pred.cpu().numpy()
        b_gt_np = b_gt.cpu().numpy()
        test_dice = dice(y_pred_np, b_gt_np, labels=(Label.BG, Label.LV, Label.MYO, Label.RV),
                         exclude_bg=True, all_classes=True)

        test_hd = hausdorff(y_pred_np, b_gt_np, labels=(Label.BG, Label.LV, Label.MYO, Label.RV),
                            exclude_bg=True, all_classes=True, voxel_spacing=voxel_spacing.cpu().numpy())

        anat_errors = is_anatomically_valid_mri(y_pred_np, voxel_spacing, is_base)

        logs = {'test_loss': loss,
                'is_anatomically_valid': anat_errors.mean(),
                'test_acc': acc.mean(),
                "test_dice": simple_dice.mean(),
                }
        logs.update(test_dice)
        logs.update(test_hd)

        for i in range(len(b_img)):
            im_show = b_img[i].permute((0, 2, 1)).detach().cpu().numpy()
            gt_show = b_gt[i].unsqueeze(0).permute((0, 2, 1)).detach().cpu().numpy()
            pred_show = y_pred[i].unsqueeze(0).permute((0, 2, 1)).detach().cpu().numpy()
            log_case(self.logger, img=[im_show,gt_show,pred_show], title='test_Figure_supervised', number=batch_idx * (i + 1),
                      img_text=simple_dice[i].mean())

        self.log_dict(logs, batch_size=b_img.shape[0])

        return logs

    def on_test_end(self) -> None:
        self.save()

    def save(self) -> None:
        if self.ckpt_path:
            torch.save(self.net.state_dict(), self.ckpt_path)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        b_img, b_gt, ids, inst, voxel_spacing, is_base = batch['img'], batch['gt'], batch['id'], batch['instant'], batch['vox'], batch['is_base']
        actions = self.forward(b_img)
        if self.output_shape[0] > 1:
            actions = actions.argmax(dim=1)
        else:
            actions = torch.round(actions)

        ######## NEED TO BE IMPLEMENTED
