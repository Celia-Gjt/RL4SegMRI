import os
from typing import Optional
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from vital.data.pepr.config import Label

background = Label.BG
lv = Label.LV
myo = Label.MYO
rv = Label.RV


class PEPRDataset(Dataset):
    def __init__(self, df, data_path, approx_gt_path=None, allow_real_gt=True, subset_frac=1.0,
                 available_gt=None, seed=0, test=False, class_label=None, num_views=16, *args, **kwargs):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.data_path = data_path
        self.approx_gt_path = approx_gt_path
        self.allow_real_gt = allow_real_gt
        self.test = test
        self.class_label = class_label
        self.num_views = num_views

        print(f"Test step: {self.test} , len of dataset {len(self.df)}")

        # pass down list of available ground truths
        self.use_gt = available_gt.fillna(False).to_numpy() if available_gt is not None else \
            np.asarray([False for _ in range(len(self.df))])
        print(f"Number of ground truths available: {self.use_gt.sum()}")

    def __len__(self):
        return len(self.df.index)

    def get_slice_position(self, z, apex_slice, base_slice):
        """
        Encode la position d'une coupe entre l'apex et la base.

        Returns:
            int: indice entre 0 et num_positions - 1
        """
        if apex_slice == base_slice:
            return self.num_views // 2

        # Position relative entre l'apex (0) et la base (1)
        position = (z - apex_slice) / (base_slice - apex_slice)

        if position < 0:
            position_idx = 0
        elif position > 1:
            position_idx = self.num_views - 1
        else:
            # Conversion en entier
            position_idx = 1 + round(position * (self.num_views - 3))

        return int(position_idx)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sub_path = row['id'] + '.nii.gz'
        z, t = int(row['slice']), int(row['instant'])
        Z, T = int(row['TOT_slice']), int(row['TOT_instant'])
        approx_gt_subpath = f"{row['id']}_{t}_{z}.nii.gz"

        img_nifti = nib.load(self.data_path + '/img/' + sub_path)

        if img_nifti.ndim == 4:
            img = np.asarray(img_nifti.dataobj[:, :, z, t])
        else:  # ndim == 3
            img = np.asarray(img_nifti.dataobj[:, :, z])

        img = np.expand_dims(img, 0)  # (1, 256, 256)
        img = img.astype(np.float32)
        # exemple de normalisation
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        if self.allow_real_gt or self.test: #) and self.use_gt[idx]:
            gt_nifti = nib.load(self.data_path + '/gt/' + sub_path)
            if gt_nifti.ndim == 4:
                gt = np.asarray(gt_nifti.dataobj[:, :, z, t])
            else:  # ndim == 3, cas Ds : gt correspond au seul instant annoté
                gt = np.asarray(gt_nifti.dataobj[:, :, z])
        else:
            gt = np.zeros(img.shape[1:])

        if self.class_label:
            gt = (gt == self.class_label)

        if self.use_gt[idx] and self.approx_gt_path is not None:
            approx_gt_path = self.approx_gt_path + '/approx_gt/' + approx_gt_subpath
            approx_gt_nifti = nib.load(approx_gt_path)
            approx_gt = np.asarray(approx_gt_nifti.dataobj[:, :])

        else:
            approx_gt = np.zeros_like(gt)

        return {'img': torch.tensor(img, dtype=torch.float32),
                'gt': torch.tensor(gt).type(torch.LongTensor) if self.allow_real_gt or self.test else torch.zeros_like(
                    torch.tensor(gt)).type(torch.LongTensor),
                'approx_gt': torch.tensor(approx_gt).type(torch.LongTensor),
                'use_gt': torch.tensor(self.use_gt[idx]),
                'id': row['id'],
                'slice': z,
                'instant': t,
                'TOT_slice': Z,
                'TOT_instant': T,
                "apex_slice": row['apex_slice'],
                "base_slice": row['base_slice'],
                "is_base": row['is_base'],
                "view_as_id": self.get_slice_position(z, row['apex_slice'], row['base_slice']),
                'vox': img_nifti.header['pixdim'][1:3]
                }


class PEPRDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self,
                 data_dir,
                 csv_file,
                 approx_gt_dir=None,
                 supervised=False,
                 gt_column=None,
                 splits_column='split_0',
                 subset_frac=1.0,
                 test_frac=0.1,
                 predict_column='pred',
                 gt_frac=None,
                 seed=0,
                 class_label=None,
                 num_views=16,
                 *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.save_hyperparameters(logger=False)
        self.num_views = num_views

        # open dataframe for dataset
        self.df_path = self.hparams.data_dir + '/' + self.hparams.csv_file + '.csv'
        self.df = pd.read_csv(self.df_path, index_col=0)

        self.df_slice_path = self.hparams.data_dir + '/' + self.hparams.csv_file + '_slice.csv'

        self.train: Optional[torch.utils.Dataset] = None
        self.validate: Optional[torch.utils.Dataset] = None
        self.test: Optional[torch.utils.Dataset] = None
        self.pred: Optional[torch.utils.Dataset] = None

    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally.
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass

    def prepare_data_per_node(self):
        pass

    def get_apex_base(self, gt_volume):
        segmented = np.any(gt_volume > 0, axis=(0, 1))
        segmented_indices = np.where(segmented)[0]

        if len(segmented_indices) == 0:
            return 0, 0

        first_slice = segmented_indices[0]
        last_slice = segmented_indices[-1]

        volume_first = gt_volume[:, :, first_slice]
        volume_last = gt_volume[:, :, last_slice]
        mask_first = (volume_first != 0) & (volume_first != rv)
        mask_last = (volume_last != 0) & (volume_last != rv)
        count_first_slice = np.count_nonzero(mask_first)
        count_last_slice = np.count_nonzero(mask_last)

        if count_first_slice < count_last_slice:
            apex_slice = first_slice
            base_slice = last_slice
        else:
            apex_slice = last_slice
            base_slice = first_slice

        return apex_slice, base_slice

    def build_slice_index(self, df, save_path):
        """
        Explose un DataFrame patient-level en DataFrame slice-level.
        Gère les patients 3D (un seul instant) et 4D (cine complet).

        Calcule également les informations apex / base à partir de la segmentation disponible.

        Si un slice-level CSV existe déjà, il est conservé tel quel : aucune coupe n'est ajoutée ou supprimée.
        Mais si les colonnes apex_slice, base_slice et is_base sont absentes, elles sont ajoutées.

        Sauvegarde le résultat à save_path."""

        if os.path.isfile(save_path) and os.path.getsize(save_path) > 0:
            result = pd.read_csv(save_path)

            required_columns = {
                'apex_slice',
                'base_slice',
                'is_base'
            }

            missing_columns = required_columns - set(result.columns)

            # Toutes les colonnes sont déjà présentes
            if not missing_columns:
                return result

            print(
                f"Colonnes manquantes dans {save_path}: "
                f"{missing_columns}"
            )
            print("Ajout des métadonnées sans modifier les slices existantes.")

            # --------------------------------------------------------
            # Calcul des apex/base uniquement pour les couples
            # (patient, instant) présents dans le CSV
            # --------------------------------------------------------

            apex_base_cache = {}

            for patient_id, instant in result[['id', 'instant']].drop_duplicates().itertuples(index=False):

                key = (patient_id, int(instant))

                gt_path = os.path.join(
                    self.hparams.data_dir,
                    'gt',
                    patient_id + '.nii.gz'
                )

                unet_path = os.path.join(
                    self.hparams.data_dir,
                    'unet',
                    patient_id + '.nii.gz'
                )

                if os.path.isfile(gt_path):
                    seg_path = gt_path
                elif os.path.isfile(unet_path):
                    seg_path = unet_path
                else:
                    raise ValueError(
                        f"Pas de segmentation pour {patient_id}"
                    )

                seg_nifti = nib.load(seg_path)

                if seg_nifti.ndim == 4:
                    gt_volume = np.asarray(
                        seg_nifti.dataobj[:, :, :, int(instant)]
                    )
                else:
                    gt_volume = np.asarray(
                        seg_nifti.dataobj[:, :, :]
                    )

                apex_slice, base_slice = self.get_apex_base(
                    gt_volume
                )

                apex_base_cache[key] = (
                    apex_slice,
                    base_slice
                )

            # --------------------------------------------------------
            # Ajouter les informations aux lignes EXISTANTES
            # --------------------------------------------------------

            apex_values = []
            base_values = []
            is_base_values = []

            for patient_id, instant, slice_idx in result[
                ['id', 'instant', 'slice']
            ].itertuples(index=False):
                apex_slice, base_slice = apex_base_cache[
                    (patient_id, int(instant))
                ]

                apex_values.append(apex_slice)
                base_values.append(base_slice)
                is_base_values.append(
                    abs(int(slice_idx) - base_slice) <= 2
                )

            result['apex_slice'] = apex_values
            result['base_slice'] = base_values
            result['is_base'] = is_base_values

            # Sauvegarde
            result.to_csv(
                save_path,
                index=False
            )

            return result

        rows = []

        for _, row in df.iterrows():
            patient_id = row['id']

            # ---------------------------------------------------------
            # IMAGE
            # ---------------------------------------------------------
            img_path = os.path.join(
                self.hparams.data_dir,
                'img',
                patient_id + '.nii.gz'
            )

            img_nifti = nib.load(img_path)
            shape = img_nifti.shape

            # ---------------------------------------------------------
            # SEGMENTATION
            # ---------------------------------------------------------

            gt_path = os.path.join(
                self.hparams.data_dir,
                'gt',
                patient_id + '.nii.gz'
            )
            unet_path = os.path.join(
                self.hparams.data_dir,
                'unet',
                patient_id + '.nii.gz'
            )

            if os.path.isfile(gt_path):
                gt_nifti = nib.load(gt_path)
            elif os.path.isfile(unet_path):
                gt_nifti = nib.load(unet_path)
            else:
                raise ValueError(
                    f"Pas de segmentation pour {patient_id}"
                )

            if len(shape) == 4:
                Z, T = shape[2], shape[3]
                for t in range(T):
                    gt_volume = np.asarray(gt_nifti.dataobj[:, :, :, t])
                    apex_slice, base_slice = self.get_apex_base(gt_volume)

                    for z in range(Z):
                        r = row.to_dict()

                        r['slice'] = z
                        r['TOT_slice'] = Z
                        r['instant'] = t
                        r['TOT_instant'] = T

                        r['apex_slice'] = apex_slice
                        r['base_slice'] = base_slice
                        r['is_base'] = abs(z - base_slice) <= 2

                        rows.append(r)

            elif len(shape) == 3:
                Z = shape[2]

                gt_volume = np.asarray(gt_nifti.dataobj[:, :, :])
                apex_slice, base_slice = self.get_apex_base(gt_volume)

                for z in range(Z):
                    r = row.to_dict()

                    r['slice'] = z
                    r['TOT_slice'] = Z
                    r['instant'] = 0  # convention : pas de dimension temporelle
                    r['TOT_instant'] = 0

                    r['apex_slice'] = apex_slice
                    r['base_slice'] = base_slice
                    r['is_base'] = abs(z - base_slice) <= 2

                    rows.append(r)

            else:
                raise ValueError(f"Shape inattendue pour {row['id']}: {shape}")
        result = pd.DataFrame(rows)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.to_csv(save_path, index=False)

        return result

    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """

        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here
        self.df_slice = self.build_slice_index(self.df, self.df_slice_path)

        # Do splits
        if self.hparams.splits_column in self.df_slice.columns:
            self.train_idx = self.df_slice.index[self.df_slice[self.hparams.splits_column] == 'train'].tolist()
            self.val_idx = self.df_slice.index[self.df_slice[self.hparams.splits_column] == 'val'].tolist()
            self.test_idx = self.df_slice.index[self.df_slice[self.hparams.splits_column] == 'test'].tolist()
            self.pred_idx = self.df_slice.index[
                self.df_slice[self.hparams.splits_column] == self.hparams.predict_column].tolist()
        else:
            # create new splits, save if column name is given
            print(f"Creating new splits!")
            train_idx, val_and_test_idx = train_test_split(self.df.index.to_list(),
                                                                train_size=0.8,
                                                                random_state=self.hparams.seed)
            val_idx, test_idx = train_test_split(val_and_test_idx,
                                                           test_size=0.5,
                                                           random_state=self.hparams.seed)
            pred_idx = []

            split_col = self.hparams.splits_column if self.hparams.splits_column else '_tmp_split'

            self.df_slice[split_col] = None
            self.df_slice.loc[self.df_slice['id'].isin(train_idx), split_col] = 'train'
            self.df_slice.loc[self.df_slice['id'].isin(val_idx), split_col] = 'val'
            self.df_slice.loc[self.df_slice['id'].isin(test_idx), split_col] = 'test'

            self.train_idx = self.df_slice.index[self.df_slice[split_col] == 'train'].tolist()
            self.val_idx = self.df_slice.index[self.df_slice[split_col] == 'val'].tolist()
            self.test_idx = self.df_slice.index[self.df_slice[split_col] == 'test'].tolist()
            self.pred_idx = []

            if self.hparams.splits_column:
                print(f"Saving new split to column: {self.hparams.splits_column}")
                self.df_slice.to_csv(self.df_slice_path, index=False)
            else:
                self.df_slice.drop(columns=[split_col], inplace=True) if split_col == '_tmp_split' else None

        if self.hparams.gt_frac:
            self.hparams.gt_column = 'Gt_0'
            patient_ids = self.df_slice['id'].unique()
            n_gt_patients = int(self.hparams.gt_frac * len(patient_ids))
            gt_patient_ids = np.random.choice(patient_ids, size=n_gt_patients, replace=False)
            self.df_slice[self.hparams.gt_column] = self.df_slice['id'].isin(gt_patient_ids)

        if self.hparams.subset_frac and type(self.hparams.subset_frac) == float:
            train_num = int(self.hparams.subset_frac * len(self.train_idx))
            self.train_idx = self.train_idx[:train_num]
            val_num = int(self.hparams.subset_frac * len(self.val_idx))
            self.val_idx = self.val_idx[:val_num]
            test_num = int(self.hparams.subset_frac * len(self.test_idx))
            self.test_idx = self.test_idx[:test_num]
            pred_num = int(self.hparams.subset_frac * len(self.pred_idx))
            self.pred_idx = self.pred_idx[-pred_num:]

        elif self.hparams.subset_frac and type(self.hparams.subset_frac) == int:
            self.train_idx = self.train_idx[:min(self.hparams.subset_frac, len(self.train_idx))]
            self.val_idx = self.val_idx[:min(self.hparams.subset_frac, len(self.val_idx))]
            self.test_idx = self.test_idx[:min(self.hparams.subset_frac, len(self.test_idx))]
            self.pred_idx = self.pred_idx[:min(self.hparams.subset_frac, len(self.pred_idx))]

        if self.hparams.approx_gt_dir is not None:
            os.makedirs(self.hparams.approx_gt_dir, exist_ok=True)

        if stage == "fit" or stage is None:
            train_slices = self.df_slice.loc[self.train_idx]
            def pick_instant(g):
                T = g['TOT_instant'].iloc[0]
                # instant tiré dans [0, T[ ; si T == 0 on garde 0V
                t = 0 if T == 0 else random.randrange(T)
                return g[g['instant'] == t]

            train_slices = (
                train_slices
                .groupby('id', group_keys=False)
                .apply(pick_instant)
                )
            self.train = PEPRDataset(train_slices,
                                     data_path=self.hparams.data_dir,
                                     subset_frac=self.hparams.subset_frac,
                                     seed=self.hparams.seed,
                                     available_gt=train_slices.get(self.hparams.gt_column, None),
                                     approx_gt_path=self.hparams.approx_gt_dir,
                                     allow_real_gt=self.hparams.supervised,
                                     class_label=self.hparams.class_label,
                                     num_views=self.num_views)

            val_slices = self.df_slice.loc[self.val_idx]

            self.validate = PEPRDataset(val_slices,
                                        data_path=self.hparams.data_dir,
                                        subset_frac=self.hparams.subset_frac,
                                        seed=self.hparams.seed,
                                        available_gt=val_slices.get(self.hparams.gt_column, None),
                                        allow_real_gt=self.hparams.supervised,
                                        class_label=self.hparams.class_label,
                                        num_views=self.num_views)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_slices = self.df_slice.loc[self.test_idx]
            self.test = PEPRDataset(test_slices,
                                    data_path=self.hparams.data_dir,
                                    subset_frac=self.hparams.subset_frac,
                                    seed=self.hparams.seed,
                                    available_gt=None,
                                    test=True,
                                    allow_real_gt=self.hparams.supervised,
                                    class_label=self.hparams.class_label,
                                    num_views=self.num_views)
        if stage == "predict":
            pred_slices = self.df_slice.loc[self.pred_idx]
            self.pred = PEPRDataset(pred_slices,
                                    data_path=self.hparams.data_dir,
                                    subset_frac=self.hparams.subset_frac,
                                    seed=self.hparams.seed,
                                    available_gt=None,
                                    test=False,
                                    allow_real_gt=False, #self.hparams.supervised,
                                    class_label=self.hparams.class_label,
                                    num_views=self.num_views)

    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=32, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=32, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=32, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.pred, batch_size=32, num_workers=4)

    def add_to_train(self, id, instant, slice):
        if torch.is_tensor(id):
            id = id.item() if id.ndim == 0 else id.cpu().numpy()

        if torch.is_tensor(instant):
            instant = instant.item()

        if torch.is_tensor(slice):
            slice = slice.item()

        self.df_slice.loc[(self.df_slice['id'] == id) &
                          (self.df_slice['slice'] == slice) &
                          (self.df_slice['instant'] == instant), self.hparams.splits_column] = 'train'

    def add_to_gt(self, id, instant, slice):
        if torch.is_tensor(id):
            id = id.item() if id.ndim == 0 else id.cpu().numpy()

        if torch.is_tensor(instant):
            instant = instant.item()

        if torch.is_tensor(slice):
            slice = slice.item()

        self.df_slice.loc[(self.df_slice['id'] == id) &
                          (self.df_slice['slice'] == slice) &
                          (self.df_slice['instant'] == instant), self.hparams.gt_column] = True

    def update_dataframe(self):
        self.df_slice.to_csv(self.df_slice_path, index=False)

    def get_approx_gt_subpath(self, id, instant, slice):
        if torch.is_tensor(id):
            id = id.item() if id.ndim == 0 else id.cpu().numpy()

        if torch.is_tensor(instant):
            instant = instant.item()

        if torch.is_tensor(slice):
            slice = slice.item()

        return f"{id}_{instant}_{slice}.nii.gz"


if __name__ == "__main__":
    dl = PEPRDataModule(data_dir='C:/Users/goujat/Documents/thesis/RL4Seg/data/pepr_cine_train',
                        csv_file='pepr',
                        subset_frac=0.1,
                        splits_column="split_0",
                        test_frac=0.1)

    dl.setup()
    for i in range(1):
        batch = next(iter(dl.train_dataloader()))
        print(batch['img'].shape)
        print(batch['gt'].shape)
        print(batch['approx_gt'].shape)
