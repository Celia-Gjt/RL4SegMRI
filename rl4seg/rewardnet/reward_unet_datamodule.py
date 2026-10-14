import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader


def group_key(path: str):
    """Identifie l'image source à partir du nom de fichier
    {batch_idx}_{itr}_{i}_{degrad_type}_{timestamp}.nii.gz"""
    batch_idx, itr, i = Path(path).name.split("_")[:3]
    return (batch_idx, i)


def split_files(data_path, test_frac=0.1, seed=1):
    files = [f.as_posix() for f in Path(data_path).rglob("**/images/*.nii.gz")]

    groups = defaultdict(list)
    for f in files:
        groups[group_key(f)].append(f)

    keys = sorted(groups)  # sorted => déterministe
    random.Random(seed).shuffle(keys)

    n_test = int(test_frac * len(keys))
    n_val = int(0.1 * len(keys))
    test_keys = keys[:n_test]
    val_keys = keys[n_test:n_test + n_val]
    train_keys = keys[n_test + n_val:]

    flat = lambda ks: [f for k in ks for f in groups[k]]
    return {"train": flat(train_keys), "val": flat(val_keys), "test": flat(test_keys)}


class RewardNetAutoDataset(Dataset):
    """ Works with the output of 'utils.file_utils.save_batch_to_dataset """

    def __init__(self, data_path, split, test_frac=0.1, seed=1):
        self.img_list = split_files(data_path, test_frac, seed)[split]
        print(f"LEN of reward net dataset ({split}): {len(self.img_list)}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = np.squeeze(nib.load(self.img_list[idx]).get_fdata())
        gt = np.squeeze(nib.load(self.img_list[idx].replace("images", "gt")).get_fdata())
        pred = np.squeeze(nib.load(self.img_list[idx].replace("images", "pred")).get_fdata())

        x = np.stack([img, pred], axis=0)  # (2, 256, 256)
        y = (gt == pred).astype(np.float32)  # (256, 256)
        y = np.expand_dims(y, axis=0)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class RewardNetDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self, data_path, *args, **kwargs):
        super().__init__()
        self.data_path = data_path

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def prepare_data_per_node(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here

        if stage == "fit" or stage is None:
            self.train = RewardNetAutoDataset(self.data_path, split="train")
            self.validate = RewardNetAutoDataset(self.data_path, split="val")
        if stage == "test" or stage is None:
            self.test = RewardNetAutoDataset(self.data_path, split="test")

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=8, num_workers=8)


if __name__ == "__main__":

    dl = RewardNetDataModule('../../logs/run_pepr_to_pepr')
    dl.setup()
    for i in range(1):
        batch = next(iter(dl.train_dataloader()))
        print(batch[0].shape)
        print(batch[1].shape)

        from matplotlib import pyplot as plt

        plt.figure()
        plt.imshow(batch[0][0, 0, :, :].cpu().numpy())

        plt.figure()
        plt.imshow(batch[0][0, 1, :, :].cpu().numpy())

        plt.figure()
        plt.imshow(batch[1][0, 0, :, :].cpu().numpy())
        plt.show()
