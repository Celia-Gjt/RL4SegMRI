import torch
from vital.metrics.camus.anatomical.utils import check_segmentation_validity
from vital.metrics.train.functional import differentiable_dice_score
from rl4seg.anatomical_metrics import check_segmentation_validity as check_segmentation_validity_mri


def accuracy(pred, imgs, gt):
    actions = torch.round(pred)
    assert actions.shape == gt.shape, \
        print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")
    simple = (actions == gt).float()
    return simple.mean(dim=(1, 2), keepdim=True)


def error_recall(pred, gt):
    """Parmi les pixels réellement mal segmentés par l'actor, combien le reward model détecte ?"""
    actions = torch.round(pred)
    assert actions.shape == gt.shape, \
        print(f"Actions shape {actions.shape} vs GT shape {gt.shape}")

    is_error = (gt == 0)
    detected = (actions == 0) & is_error

    reduce_dims = tuple(range(1, gt.ndim))  # toutes les dims sauf batch
    return detected.float().sum(dim=reduce_dims) / is_error.float().sum(dim=reduce_dims).clamp(min=1)


def dice_score(output, target, num_classes=4):
    out = torch.zeros(output.shape[0], device=output.device)

    for i in range(output.shape[0]):
        dices = []

        for c in range(1, num_classes):  # ignore background
            target_c = (target[i] == c)
            output_c = (output[i] == c)

            target_sum = target_c.sum()
            output_sum = output_c.sum()

            if target_sum == 0 and output_sum == 0:
                continue

            intersection = (target_c & output_c).sum()

            dice_c = (2.0 * intersection) / (target_sum + output_sum)
            dices.append(dice_c)

        out[i] = torch.stack(dices).mean() if dices else 1.0

    return out


def is_anatomically_valid(output):
    out = torch.zeros(len(output))
    for i in range(len(output)):
        try:
            out[i] = int(check_segmentation_validity(output[i].T, (1.0, 1.0), [0, 1, 2]))
        except:
            out[i] = 0
    return out


def is_anatomically_valid_mri(output, voxelspacing, is_base):
    out = torch.zeros(len(output))

    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()

    if isinstance(voxelspacing, torch.Tensor):
        voxelspacing = voxelspacing.detach().cpu().numpy()

    if isinstance(is_base, torch.Tensor):
        is_base = is_base.detach().cpu().numpy()

    for i in range(len(output)):
        out[i] = int(check_segmentation_validity_mri(output[i], voxelspacing[i], is_base[i])[0])
    return out
