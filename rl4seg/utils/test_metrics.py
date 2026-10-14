from typing import Tuple

import numpy as np
from medpy.metric import dc, hd
from vital.data.config import LabelEnum


def dice(pred: np.ndarray,
         target: np.ndarray,
         labels: Tuple[LabelEnum],
         exclude_bg: bool = True,
         all_classes: bool = False):
    """Compute dice for one sample.

    Args:
        pred: prediction array in categorical form (H, W)
        target: target array in categorical form (H, W)
        labels: List of labels for which to compute the dice.
        exclude_bg: If true, background dice is not considered.

    Returns:
        mean dice
    """
    dices = []
    dice_dict = {}

    for label in labels:
        if exclude_bg and label == 0:
            continue

        pred_mask = np.isin(pred, label)
        gt_mask = np.isin(target, label)

        d = dc(pred_mask, gt_mask)
        dices.append(d)

        if all_classes:
            dice_dict[f"dice_{label.name}"] = d

    mean_dice = np.mean(dices)

    if all_classes:
        dice_dict["Dice"] = mean_dice
        return dice_dict

    return mean_dice


def hausdorff(pred: np.ndarray, target: np.ndarray, labels: Tuple[LabelEnum], exclude_bg: bool = True,
              all_classes: bool = False, voxel_spacing: Tuple[float] = None):
    """Compute hausdorff for one sample.

    Args:
        pred: prediction array in categorical form (H, W)
        target: target array in categorical form (H, W)
        labels: List of labels for which to compute the metric.
        exclude_bg: If true, background dice is not considered.

    Returns:
        hausdorff
    """
    raw_values = {}  # clé -> liste de valeurs (potentiellement nan) sur tous les samples

    for i in range(len(pred)):
        hausdorffs = []

        for label in labels:
            if exclude_bg and label == 0:
                continue

            pred_mask = pred[i] == label
            gt_mask = target[i] == label

            pred_empty = not np.any(pred_mask)
            gt_empty = not np.any(gt_mask)

            if pred_empty and gt_empty:
                haus = 0.0
            elif pred_empty or gt_empty:
                haus = np.nan
            else:
                haus = hd(
                    pred_mask,
                    gt_mask,
                    voxel_spacing[i] if voxel_spacing is not None else None
                )

            hausdorffs.append((label, haus))

        if not hausdorffs:
            continue

        if all_classes:
            for label, haus in hausdorffs:
                key = f"hd_{label.name}"
                raw_values.setdefault(key, []).append(haus)

        hd_values = [haus for _, haus in hausdorffs]
        if all(np.isnan(hd_values)):
            sample_hd = np.nan
        else:
            sample_hd = np.nanmean(hd_values)
        raw_values.setdefault('Hausdorff', []).append(sample_hd)

    # Agrégation finale, nan-safe, une seule fois
    hd_dict = {}
    for key, values in raw_values.items():
        if all(np.isnan(v) for v in values):
            hd_dict[key] = np.nan
        else:
            hd_dict[key] = np.nanmean(values)

    return hd_dict
