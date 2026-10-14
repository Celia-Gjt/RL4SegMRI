import os
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import torch


def get_img_subpath(row, suffix='', extension='.nii.gz'):
    """
    Format string for partial path of image in file structure
    :param row: dataframe row with all columns filled in
    :param suffix: suffix before file extension
    :param extension: file extension
    :return: string containing path to image file
    """
    return f"{row['study']}/{row['view'].lower()}/{row['dicom_uuid']}" + suffix + extension


def save_to_reward_dataset(save_dir, filename, image, gt, action, spacing):
    for sub in ("images", "gt", "pred"):
        Path(f"{save_dir}/{sub}").mkdir(parents=True, exist_ok=True)

    def to_numpy(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.squeeze(x)

    image, gt, action = to_numpy(image), to_numpy(gt), to_numpy(action)
    if torch.is_tensor(spacing):
        spacing = spacing.numpy()
    sx, sy = float(spacing[0]), float(spacing[1])

    # affine diagonale correcte
    affine = np.diag([-sx, -sy, 1.0, 1.0])

    for arr, sub, dtype in [(image, "images", np.float32),
                            (gt, "gt", np.uint8),
                            (action, "pred", np.uint8)]:
        nib.Nifti1Image(np.expand_dims(arr, axis=-1).astype(dtype), affine).to_filename(f"{save_dir}/{sub}/{filename}")


def subfiles(
        folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True
) -> List[str]:
    """This function returns a list of file names from a specified folder.

    Args:
        folder: The folder path to search for files.
        join: If True, the folder path is joined with the file names. Otherwise, only file names are returned.
        prefix: Optional; filters files that start with this prefix.
        suffix: Optional; filters files that end with this suffix.
        sort: If True, the resulting file list is sorted.

    Returns:
        List[str]: A list of file paths or file names based on the 'join' parameter.
    """
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E731, E741
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res
