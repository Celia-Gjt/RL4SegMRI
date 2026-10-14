import copy
import random
from datetime import datetime
from typing import Any

import h5py
import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from scipy import ndimage
from torchvision.transforms.functional import adjust_contrast
from torch import Tensor
from rl4seg.anatomical_metrics import check_segmentation_validity as check_segmentation_validity_mri
from rl4seg.utils.deformation_helper import punch_hole_in_structure, remove_rv, punch_hole_at_interface, \
    sample_deformation_size, erode_structure, dilate_structure, shift_interface
from vital.data.pepr.config import Label

from utils.Metrics import accuracy, dice_score, is_anatomically_valid_mri, error_recall
from utils.file_utils import get_img_subpath, save_to_reward_dataset
from utils.logging_helper import log_image, log_case
from utils.tensor_utils import convert_to_numpy
from utils.test_metrics import dice, hausdorff


def shrink_perturb(model, lamda=0.5, sigma=0.01):
    for (name, param) in model.named_parameters():
        if 'weight' in name:  # just weights
            # nc = param.shape[0]  # cols
            # nr = param.shape[1]  # rows
            # for i in range(nr):
            #     for j in range(nc):
            param.data = \
                (lamda * param.data) + \
                torch.normal(0.0, sigma, size=(param.data.shape), device=next(model.parameters()).device)
    return model


class RLmodule(pl.LightningModule):

    def __init__(self, actor, reward,
                 corrector=None,
                 actor_save_path=None,
                 critic_save_path=None,
                 save_uncertainty_path=None,
                 predict_save_dir=None,
                 predict_do_hole_degrad=True,
                 predict_do_shape_deform=True,
                 predict_do_swap_img=True,
                 predict_do_model_perturb=True,
                 predict_do_img_perturb=True,
                 predict_do_corrections=True,
                 *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        #TODO: use automatic hparams

        self.actor = actor
        self.reward_func = reward

        self.actor_save_path = actor_save_path
        self.critic_save_path = critic_save_path

        self.save_uncertainty_path = save_uncertainty_path

        self.predict_save_dir = predict_save_dir
        self.pred_corrector = corrector

        self.predict_do_hole_degrad = predict_do_hole_degrad
        self.predict_do_shape_deform = predict_do_shape_deform
        self.predict_do_swap_img = predict_do_swap_img
        self.predict_do_model_perturb = predict_do_model_perturb
        self.predict_do_img_perturb = predict_do_img_perturb
        self.predict_do_corrections = predict_do_corrections

        self.invalid_segmentation_bank = {}

    def configure_optimizers(self):
        return self.actor.get_optimizers()

    @torch.no_grad()  # no grad since tensors are reused in PPO's for loop
    def rollout(self, imgs: torch.tensor, gt: torch.tensor, use_gt: torch.tensor = None, sample: bool = True):
        """
            Rollout the policy over a batch of images and ground truth pairs
        Args:
            imgs: batch of images
            gt: batch of ground truth segmentation maps
            use_gt: replace policy result with ground truth (bool mask of len of batch)
            sample: whether to sample actions from distribution or determinist

        Returns:
            Actions (used for rewards, log_pobs, etc), sampled_actions (mainly for display), log_probs, rewards
        """
        actions = self.actor.act(imgs, sample=sample)
        rewards = self.reward_func(actions, imgs, gt)

        if use_gt is not None:
            actions[use_gt, ...] = gt[use_gt, ...]

        _, _, log_probs, _, _, _ = self.actor.evaluate(imgs, actions)
        return actions, log_probs, rewards

    def training_step(self, batch: dict[str, Tensor], nb_batch):
        """
            Defines training steo, calculates loss based on the minibatch recieved.
            Optionally backprop through networks with self.automatic_optimization = False flag,
            otherwise return loss dict and Lighting does it automatically
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics or None
        """
        raise NotImplementedError

    def compute_policy_loss(self, batch, sample=True, **kwargs):
        """
            Compute unsupervised loss to maximise reward using policy gradient method.
        Args:
            batch: batch of images, actions, log_probs, rewards and groudn truth
            sample: whether to sample from distribution or deterministic approach (mainly for val, test steps)

        Returns:
            mean loss(es) for the batch, metrics dictionary
        """
        raise NotImplementedError

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int):
        """
            Defines validation step (using sampling to show model confidence)
            Computes actions from current policy and calculates loss, rewards (and other metrics)
            Logs images and segmentations to logger
        Args:
            batch: batch of images and ground truth
            batch_idx: index of batch

        Returns:
            Dict of logs
        """
        b_img, b_gt, b_use_gt, voxel_spacing, is_base = batch['img'], batch['gt'], batch['use_gt'], batch['vox'], batch[
            'is_base']

        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, sample=False)

        loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                    prev_log_probs, b_gt, b_use_gt))

        acc = accuracy(prev_actions, b_img, b_gt)
        dice = dice_score(prev_actions, b_gt, self.actor.actor.output_shape[0])
        anat_errors = is_anatomically_valid_mri(prev_actions, voxel_spacing, is_base)

        logs = {'val_loss': loss,
                "val_reward": torch.mean(prev_rewards.type(torch.float)),
                "val_acc": acc.mean(),
                "val_dice": dice.mean(),
                "val_anat_valid": anat_errors.mean(),
                }

        # log images
        idx = random.randint(0, len(b_img) - 1)  # which image to log
        im_show = b_img[idx].permute((0, 2, 1)).detach().cpu().numpy()
        gt_show = b_gt[idx].unsqueeze(0).permute((0, 2, 1)).detach().cpu().numpy()
        pred_show = prev_actions[idx].unsqueeze(0).permute((0, 2, 1)).detach().cpu().numpy()
        if prev_rewards.shape == prev_actions.shape:
            reward_show = prev_rewards[idx].unsqueeze(0).permute((0, 2, 1)).detach().cpu().numpy()
            # log_image(self.logger, img=reward_show, title='RewardMap_RL', number=self.global_step)
            log_case(self.logger, img=[im_show, gt_show, pred_show, reward_show], title='Figure_RL',
                     number=self.global_step,
                     img_text=prev_rewards[idx].mean())
        else:
            log_case(self.logger, img=[im_show, gt_show, pred_show], title='Figure_RL', number=self.global_step,
                     img_text=prev_rewards[idx].mean())

        self.log_dict(logs)
        return logs

    def test_step(self, batch: dict[str, Tensor], batch_idx: int):
        """
            Defines test step (uses deterministic method to show real results)
            Computes actions from current policy and calculates loss, rewards (and other metrics)
            Logs images and segmentations to logger
        Args:
            batch: batch of images and ground truth
            batch_idx: index of batch

        Returns:
            Dict of logs
        """
        b_img, b_gt, voxel_spacing, is_base = batch['img'], batch['gt'], batch['vox'], batch['is_base']

        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_gt, sample=False)
        loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                    prev_log_probs, b_gt, False))
        acc = accuracy(prev_actions, b_img, b_gt)
        simple_dice = dice_score(prev_actions, b_gt, self.actor.actor.output_shape[0])
        y_pred_np = prev_actions.cpu().numpy()
        b_gt_np = b_gt.cpu().numpy()

        for i in range(len(y_pred_np)):
            lbl, num = ndimage.measurements.label(y_pred_np[i] != 0)

            # No foreground predicted
            if num == 0:
                continue

            # Count the number of elements per label
            count = np.bincount(lbl.flat)
            # Select the largest blob
            maxi = np.argmax(count[1:]) + 1
            # Remove the other blobs
            y_pred_np[i][lbl != maxi] = 0

        test_dice = dice(y_pred_np, b_gt_np, labels=(Label.BG, Label.LV, Label.MYO, Label.RV),
                         exclude_bg=True, all_classes=True)

        test_hd = hausdorff(y_pred_np, b_gt_np, labels=(Label.BG, Label.LV, Label.MYO, Label.RV),
                            exclude_bg=True, all_classes=True, voxel_spacing=voxel_spacing.cpu().numpy())

        anat_errors = is_anatomically_valid_mri(y_pred_np, voxel_spacing, is_base)

        logs = {'test_loss': loss,
                "test_anat_valid": anat_errors.mean(),
                "test_reward": torch.mean(prev_rewards.type(torch.float)),
                'test_acc': acc.mean(),
                "test_dice": simple_dice.mean(),
                }
        logs.update(test_dice)
        logs.update(test_hd)

        # for logging v
        _, _, _, _, v, _ = self.actor.evaluate(b_img, prev_actions)

        for i in range(len(b_img)):

            im_show = b_img[i].permute((0, 2, 1)).detach().cpu().numpy()
            gt_show = b_gt[i].unsqueeze(0).permute((0, 2, 1)).detach().cpu().numpy()
            pred_show = prev_actions[i].unsqueeze(0).permute((0, 2, 1)).detach().cpu().numpy()

            if v.shape == prev_actions.shape:
                v_show = v[i].unsqueeze(0).permute((0, 2, 1)).detach().cpu().numpy()
                # log_image(self.logger, img=v_show, title='test_v_function_RL', number=batch_idx * (i + 1), img_text=v[i].mean())
                if prev_rewards.shape == prev_actions.shape:
                    reward_show = prev_rewards[i].unsqueeze(0).permute((0, 2, 1))
                    # log_image(self.logger, img=reward_show, title='test_RewardMap_RL', number=batch_idx * (i + 1), img_text=prev_rewards[i].mean())
                    log_case(self.logger, img=[im_show, gt_show, pred_show, reward_show, v_show],
                             title='test_Figure_RL', number=batch_idx * (i + 1),
                             img_text=f"{prev_rewards[i].mean()} et {v[i].mean()}")
                else:
                    log_case(self.logger, img=[im_show, gt_show, pred_show, v_show], title='test_Figure_RL',
                             number=batch_idx * (i + 1),
                             img_text=v[i].mean())
            else:
                if prev_rewards.shape == prev_actions.shape:
                    reward_show = prev_rewards[i].unsqueeze(0).permute((0, 2, 1))
                    # log_image(self.logger, img=reward_show, title='test_RewardMap_RL', number=batch_idx * (i + 1), img_text=prev_rewards[i].mean())
                    log_case(self.logger, img=[im_show, gt_show, pred_show, reward_show], title='test_Figure_RL',
                             number=batch_idx * (i + 1),
                             img_text=prev_rewards[i].mean())
                else:
                    log_case(self.logger, img=[im_show, gt_show, pred_show], title='test_Figure_RL',
                             number=batch_idx * (i + 1))

        self.log_dict(logs)

        if self.save_uncertainty_path:
            with h5py.File(self.save_uncertainty_path, 'a') as f:
                for i in range(len(b_img)):
                    dicom = batch['id'][i] + "_" + batch['instant'][i]
                    if dicom not in f:
                        f.create_group(dicom)
                    f[dicom]['img'] = b_img[i].cpu().numpy()
                    f[dicom]['gt'] = b_gt[i].cpu().numpy()
                    f[dicom]['pred'] = prev_actions[i].cpu().numpy()
                    f[dicom]['reward_map'] = prev_rewards[i].cpu().numpy()
                    f[dicom]['accuracy_map'] = (prev_actions[i].cpu().numpy() != b_gt[i].cpu().numpy()).astype(np.uint8)

        return logs

    def on_test_end(self) -> None:
        if self.actor_save_path:
            torch.save(self.actor.actor.net.state_dict(), self.actor_save_path)
        if self.critic_save_path:
            torch.save(self.actor.critic.net.state_dict(), self.critic_save_path)

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Any:
        b_img, ids, inst, slice, voxelspacing, is_base = batch['img'], batch['id'], batch['instant'], batch['slice'], \
            batch['vox'], batch['is_base']

        actions, _, _ = self.rollout(b_img, torch.zeros_like(b_img).squeeze(1), sample=True)
        actions_unsampled, _, _ = self.rollout(b_img, torch.zeros_like(b_img).squeeze(1), sample=False)

        if self.pred_corrector is not None:
            corrected, corrected_validity, ae_comp = self.pred_corrector.correct_batch(b_img, actions_unsampled)

        initial_params = copy.deepcopy(self.actor.actor.net.state_dict())
        itr = 0

        for i in range(len(b_img)):
            valid, _ = check_segmentation_validity_mri(
                actions_unsampled[i].cpu().numpy(), voxelspacing[i].cpu().numpy(), is_base[i].cpu().numpy()
            )
            # self.trainer.datamodule.add_to_train(ids[i], inst[i], slice[i])
            if ((self.pred_corrector is None) or ae_comp[i] > 0.95):
                if valid:
                    """self.trainer.datamodule.add_to_gt(ids[i], inst[i], slice[i])
    
                    path = self.trainer.datamodule.get_approx_gt_subpath(ids[i], inst[i], slice[i])
                    approx_gt_path = self.trainer.datamodule.hparams.approx_gt_dir + '/approx_gt/' + path
                    Path(approx_gt_path).parent.mkdir(parents=True, exist_ok=True)
                    hdr = nib.Nifti1Header()
                    nifti = nib.Nifti1Image(torch.round(actions_unsampled[i]).cpu().numpy(), np.diag(np.asarray([-1, -1, 1, 0])), hdr)
                    nifti.to_filename(approx_gt_path)"""

                    if self.predict_do_hole_degrad:

                        action_np = convert_to_numpy(
                            actions_unsampled[i]
                        )

                        deformation_types = [
                            "myo_hole",
                            "rv_hole",
                            "lv_hole",
                            "no_rv",
                            "lv_myo_interface",
                            "rv_myo_interface",
                        ]

                        for degrad_type in deformation_types:

                            # 1 chance sur 5
                            if random.random() >= 1 / 5:
                                continue

                            n_holes = random.choices(
                                [1, 2, 3],
                                weights=[0.65, 0.25, 0.10],
                                k=1,
                            )[0]

                            degraded_action = action_np.copy()

                            # ---------------------------------------------------------
                            # Internal holes
                            # ---------------------------------------------------------

                            if degrad_type == "myo_hole":

                                degraded_action = punch_hole_in_structure(
                                    action_np,
                                    target_label=Label.MYO,
                                    hole_radius_range=(1, 5),
                                    num_holes=n_holes,
                                    min_boundary_distance=2,
                                )

                            elif degrad_type == "rv_hole":

                                degraded_action = punch_hole_in_structure(
                                    action_np,
                                    target_label=Label.RV,
                                    hole_radius_range=(1, 4),
                                    num_holes=n_holes,
                                    min_boundary_distance=1,
                                )

                            elif degrad_type == "lv_hole":

                                degraded_action = punch_hole_in_structure(
                                    action_np,
                                    target_label=Label.LV,
                                    hole_radius_range=(1, 4),
                                    num_holes=n_holes,
                                    min_boundary_distance=1,
                                )

                            # ---------------------------------------------------------
                            # Missing RV
                            # ---------------------------------------------------------

                            elif degrad_type == "no_rv":

                                degraded_action = remove_rv(
                                    action_np,
                                    replacement_label=Label.BG,
                                    fraction_range=(0.2, 1),
                                )

                            # ---------------------------------------------------------
                            # Interfaces
                            # ---------------------------------------------------------

                            elif degrad_type == "lv_myo_interface":

                                degraded_action = punch_hole_at_interface(
                                    action_np,
                                    Label.LV,
                                    Label.MYO,
                                    hole_radius_range=(1, 3),
                                    num_holes=n_holes,
                                )

                            elif degrad_type == "rv_myo_interface":

                                degraded_action = punch_hole_at_interface(
                                    action_np,
                                    Label.RV,
                                    Label.MYO,
                                    hole_radius_range=(1, 3),
                                    num_holes=n_holes,
                                )

                            # ---------------------------------------------------------
                            # Save
                            # ---------------------------------------------------------

                            if not np.array_equal(
                                    degraded_action,
                                    action_np,
                            ):
                                filename = (
                                    f"{batch_idx}_{itr}_{i}_"
                                    f"{degrad_type}_"
                                    f"{int(round(datetime.now().timestamp()))}.nii.gz"
                                )

                                save_to_reward_dataset(
                                    self.predict_save_dir,
                                    filename,
                                    convert_to_numpy(b_img[i]),
                                    np.expand_dims(action_np, 0),
                                    np.expand_dims(degraded_action, 0),
                                    voxelspacing[i].cpu().numpy(),
                                )

                    """
                    if self.predict_do_swap_img:

                        slice_id = int(slice[i])

                        candidates = self.invalid_segmentation_bank.get(slice_id, [])

                        if len(candidates) > 0:
                            candidate = random.choice(candidates)

                            wrong_action = candidate["action"]

                            filename = (
                                f"{batch_idx}_{itr}_{i}_wrong_s_"
                                f"{int(round(datetime.now().timestamp()))}.nii.gz"
                            )

                            save_to_reward_dataset(
                                self.predict_save_dir,
                                filename,
                                convert_to_numpy(b_img[i]),
                                np.expand_dims(
                                    convert_to_numpy(actions_unsampled[i]),
                                    0
                                ),
                                np.expand_dims(
                                    wrong_action,
                                    0
                                ),
                                voxelspacing[i].cpu().numpy()
                            )"""

                    if self.predict_do_shape_deform:

                        action_np = convert_to_numpy(
                            actions_unsampled[i])

                        deformation_types = [
                            "lv_dilate",
                            "myo_erode",
                            "myo_dilate",
                            "rv_erode",
                            "rv_dilate",
                            "lv_myo_shift",
                            "rv_myo_shift",
                        ]

                        for degrad_type in deformation_types:

                            # 1 chance sur 6 comme dans ton code actuel
                            if random.randint(0, 5) != 0:
                                continue

                            n_iter = sample_deformation_size()

                            degraded_action = action_np.copy()

                            # -------------------------------------------------------------
                            # LV
                            # -------------------------------------------------------------

                            if degrad_type == "lv_dilate":

                                degraded_action = dilate_structure(
                                    action_np,
                                    target_label=Label.LV,
                                    allowed_labels=[Label.MYO],
                                    n_iter=n_iter,
                                    local=True,
                                )

                            # -------------------------------------------------------------
                            # MYO
                            # -------------------------------------------------------------

                            elif degrad_type == "myo_erode":

                                degraded_action = erode_structure(
                                    action_np,
                                    target_label=Label.MYO,
                                    replacement_label=Label.BG,
                                    n_iter=n_iter,
                                    local=True,
                                    contact_label=[Label.BG, Label.RV]
                                )

                            elif degrad_type == "myo_dilate":

                                degraded_action = dilate_structure(
                                    action_np,
                                    target_label=Label.MYO,
                                    allowed_labels=[Label.BG, Label.RV],
                                    n_iter=n_iter,
                                    local=True,
                                )

                            # -------------------------------------------------------------
                            # RV
                            # -------------------------------------------------------------

                            elif degrad_type == "rv_erode":

                                degraded_action = erode_structure(
                                    action_np,
                                    target_label=Label.RV,
                                    replacement_label=Label.BG,
                                    n_iter=n_iter,
                                    local=True,
                                    contact_label=[Label.BG, Label.MYO]
                                )

                            elif degrad_type == "rv_dilate":

                                degraded_action = dilate_structure(
                                    action_np,
                                    target_label=Label.RV,
                                    allowed_labels=[Label.BG, Label.MYO],
                                    n_iter=n_iter,
                                    local=True,
                                )

                            # -------------------------------------------------------------
                            # Interfaces
                            # -------------------------------------------------------------

                            elif degrad_type == "lv_myo_shift":

                                degraded_action = shift_interface(
                                    action_np,
                                    label_a=Label.LV,
                                    label_b=Label.MYO,
                                    n_iter=n_iter,
                                )

                            elif degrad_type == "rv_myo_shift":

                                degraded_action = shift_interface(
                                    action_np,
                                    label_a=Label.RV,
                                    label_b=Label.MYO,
                                    n_iter=n_iter,
                                )

                            # -------------------------------------------------------------
                            # Save
                            # -------------------------------------------------------------

                            if not np.array_equal(
                                    degraded_action,
                                    action_np,
                            ):
                                filename = (
                                    f"{batch_idx}_{itr}_{i}_"
                                    f"{degrad_type}_"
                                    f"{int(round(datetime.now().timestamp()))}.nii.gz"
                                )

                                save_to_reward_dataset(
                                    self.predict_save_dir,
                                    filename,
                                    convert_to_numpy(b_img[i]),
                                    np.expand_dims(action_np, 0),
                                    np.expand_dims(degraded_action, 0),
                                    voxelspacing[i].cpu().numpy(),
                                )

                    if self.predict_do_model_perturb:
                        for j, multiplier in enumerate([0.005, 0.01, 0.015, 0.02]):
                            # get random seed based on time to maximise randomness of noise and subsequent predictions
                            # explore as much space around policy as possible
                            time_seed = int(round(datetime.now().timestamp())) + i
                            torch.manual_seed(time_seed)

                            # load initial params so noise is not compounded
                            self.actor.actor.net.load_state_dict(initial_params, strict=True)

                            # add noise to params
                            with torch.no_grad():
                                for param in self.actor.actor.net.parameters():
                                    param.add_(torch.randn(param.size()).cuda() * multiplier)

                            # make prediction
                            deformed_action, *_ = self.actor.actor(b_img[i].unsqueeze(0))
                            if len(deformed_action.shape) > 3:
                                deformed_action = deformed_action.argmax(dim=1)
                            else:
                                deformed_action = torch.round(deformed_action)

                            if deformed_action.sum() == 0:
                                continue
                            if not np.array_equal(convert_to_numpy(actions_unsampled[i]),
                                                  convert_to_numpy(deformed_action)):
                                filename = f"{batch_idx}_{itr}_{i}_{time_seed}.nii.gz"
                                save_to_reward_dataset(self.predict_save_dir,
                                                       filename,
                                                       convert_to_numpy(b_img[i]),
                                                       np.expand_dims(convert_to_numpy(actions_unsampled[i]), 0),
                                                       convert_to_numpy(deformed_action),
                                                       voxelspacing[i].cpu().numpy())
                        self.actor.actor.net.load_state_dict(initial_params, strict=True)
                    if self.predict_do_img_perturb:
                        contrast_factors = [0.8, 0.9]  # [0.7, 0.8]
                        for factor in contrast_factors:
                            in_img = b_img[i].unsqueeze(0).clone()
                            in_img = adjust_contrast(in_img, factor)
                            # in_img /= in_img.max()

                            # make prediction
                            contr_action, *_ = self.actor.actor(in_img)
                            if len(contr_action.shape) > 3:
                                contr_action = contr_action.argmax(dim=1)
                            else:
                                contr_action = torch.round(contr_action)

                            if contr_action.sum() == 0:
                                continue

                            time_seed = int(round(datetime.now().timestamp())) + int(factor * 10)

                            if not np.array_equal(convert_to_numpy(actions_unsampled[i]),
                                                  convert_to_numpy(contr_action)):
                                filename = f"{batch_idx}_{itr}_{i}_{time_seed}_contrast.nii.gz"
                                save_to_reward_dataset(self.predict_save_dir,
                                                       filename,
                                                       convert_to_numpy(b_img[i]),
                                                       np.expand_dims(convert_to_numpy(actions_unsampled[i]), 0),
                                                       convert_to_numpy(contr_action),
                                                       voxelspacing[i].cpu().numpy())

                        gaussian_blurs = [0.05, 0.075, 0.1]  # [0.05, 0.1]
                        for blur in gaussian_blurs:
                            in_img = b_img[i].unsqueeze(0).clone()
                            in_img += torch.randn(in_img.size()).cuda() * blur
                            in_img /= in_img.max()

                            # make prediction
                            blurred_action, *_ = self.actor.actor(in_img)
                            if len(blurred_action.shape) > 3:
                                blurred_action = blurred_action.argmax(dim=1)
                            else:
                                blurred_action = torch.round(blurred_action)

                            # f, (ax1, ax2) = plt.subplots(1, 2)
                            # ax1.set_title(f"Good action")
                            # ax1.imshow(actions_unsampled[i, ...].cpu().numpy().T)
                            #
                            # ax2.set_title(f"blurred action")
                            # ax2.imshow(blurred_action[0, ...].cpu().numpy().T)
                            # plt.show()

                            if blurred_action.sum() == 0:
                                continue

                            time_seed = int(round(datetime.now().timestamp())) + int(blur * 100)
                            if not np.array_equal(convert_to_numpy(actions_unsampled[i]),
                                                  convert_to_numpy(blurred_action)):
                                filename = f"{batch_idx}_{itr}_{i}_{time_seed}_blur.nii.gz"
                                save_to_reward_dataset(self.predict_save_dir,
                                                       filename,
                                                       convert_to_numpy(b_img[i]),
                                                       np.expand_dims(convert_to_numpy(actions_unsampled[i]), 0),
                                                       convert_to_numpy(blurred_action),
                                                       voxelspacing[i].cpu().numpy())

                elif self.pred_corrector is not None:
                    if not corrected_validity[i]:
                        continue

                    # if ae_comp[i] < 0.85:
                    #     f, (ax1) = plt.subplots(1)
                    #     #ax1.set_title(f"Bad initial action")
                    #     ax1.imshow(actions[i, ...].cpu().numpy().T, cmap='gray')
                    #     ax1.axis('off')
                    #     plt.savefig("/data/bad_initial.png", bbox_inches = 'tight', pad_inches = 0)
                    #
                    #     f2, (ax2) = plt.subplots(1)
                    #     # ax2.set_title(f"Corrected action")
                    #     ax2.imshow(corrected[i, ...].T, cmap='gray')
                    #     ax2.axis('off')
                    #     plt.savefig('/data/corrected.png', bbox_inches = 'tight', pad_inches = 0)
                    #
                    #     f3, (ax3) = plt.subplots(1)
                    #     # ax2.set_title(f"Corrected action")
                    #     ax3.imshow(b_img[i, ...].cpu().numpy().T, cmap='gray')
                    #     ax3.axis('off')
                    #     plt.savefig('/data/corr_image.png', bbox_inches = 'tight', pad_inches = 0)
                    #
                    #     plt.show()
                    if self.predict_do_corrections:
                        filename = f"{batch_idx}_{itr}_{i}_{int(round(datetime.now().timestamp()))}.nii.gz"
                        save_to_reward_dataset(self.predict_save_dir,
                                               filename,
                                               convert_to_numpy(b_img[i]),
                                               convert_to_numpy(corrected[i]),
                                               np.expand_dims(convert_to_numpy(actions[i]), 0),
                                               voxelspacing[i].cpu().numpy())
                """else:
                    slice_id = int(slice[i])

                    if slice_id not in self.invalid_segmentation_bank:
                        self.invalid_segmentation_bank[slice_id] = []

                    self.invalid_segmentation_bank[slice_id].append({
                        "action": convert_to_numpy(actions_unsampled[i]).copy(),
                        "id": ids[i],
                        "instant": inst[i],
                    })"""

        self.trainer.datamodule.update_dataframe()
        # make sure initial params are back at end of step
        self.actor.actor.net.load_state_dict(initial_params)


