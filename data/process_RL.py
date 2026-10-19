import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import SimpleITK as sitk
import numpy as np
from skimage.transform import resize
import matplotlib.patches as patches

from rl4seg.utils.data_processing import calculate_new_shape
from rl4seg.utils.file_utils import subfiles

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from skimage.segmentation import find_boundaries


"""def _compute_contours(seg: np.ndarray) -> np.ndarray:
    T, Z = seg.shape[0], seg.shape[1]
    # uint8 suffit très largement pour des labels de segmentation
    # (au lieu de int32 -> /4 la mémoire utilisée par ce buffer)
    contour = np.zeros_like(seg, dtype=np.uint8)

    for t in range(T):
        for z in range(Z):
            slice_seg = seg[t, z]
            if not np.any(slice_seg):
                continue  # rien à contourer, on évite l'appel inutile
            boundary_mask = find_boundaries(slice_seg, mode="inner")
            contour[t, z] = boundary_mask * slice_seg

    return contour"""


def save_BB_as_gif(image, seg, centroid_3d, save_name, box_size=256):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.axis('off')
    ax2.axis('off')

    T, Z = image.shape[0], image.shape[1]
    frames = [(t, z) for t in range(T) for z in range(Z)]

    cmap = ListedColormap([
        (0, 0, 0, 0),
        (1, 0, 0, 1),
        (0, 1, 0, 1),
        (0, 0, 1, 1)
    ])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    # --- init display (safe) ---
    im1 = ax1.imshow(image[0, 0], cmap="gray", animated=True)
    im2 = ax2.imshow(image[0, 0], cmap="gray", animated=True)

    # --- contour volume ---
    contour = np.zeros_like(seg, dtype=np.int32)

    for label_val in np.unique(seg):
        if label_val == 0:
            continue
        mask = (seg == label_val)
        for t in range(mask.shape[0]):
            for z in range(mask.shape[1]):
                contour[t, z][find_boundaries(mask[t, z], mode="inner")] = label_val

    # --- ROI bbox (fixe ou global) ---
    rect = None
    if centroid_3d is not None:
        cy, cx = centroid_3d[1:]

        r_min, r_max, c_min, c_max = get_roi_box(
            (cy, cx),
            box_size,
            image.shape[2:]
        )

        rect = patches.Rectangle(
            (c_min, r_min),
            c_max - c_min,
            r_max - r_min,
            linewidth=2,
            edgecolor='cyan',
            facecolor='none',
            animated=True
        )
        ax2.add_patch(rect)

    # --- initial contour ---
    cont = ax2.imshow(contour[0, 0], cmap=cmap, norm=norm,
                      interpolation="none", animated=True)

    # --- animation ---
    def update(idx):
        t, z = frames[idx]

        im1.set_array(image[t, z])
        im2.set_array(image[t, z])
        cont.set_array(contour[t, z])

        ax1.set_title(f"Image t={t}, z={z}")
        ax2.set_title(f"Image + seg t={t}, z={z}")

        return im1, im2, cont

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=200,
        blit=True
    )

    ani.save(f"{save_name}.gif", writer="pillow")
    plt.close()


def scale_intensity(x, lower=0.5, upper=95, b_min=-1, b_max=1):
    x = x.astype(np.float32)

    p1 = np.percentile(x, lower)
    p2 = np.percentile(x, upper)

    if p2 - p1 < 1e-8:
        return np.clip(x, b_min, b_max)

    x = np.clip(x, p1, p2)
    x = (x - p1) / (p2 - p1)
    x = x * (b_max - b_min) + b_min
    return x


def get_roi_box(centroid, box_size, img_shape):
    """
    Retourne (row_min, row_max, col_min, col_max) d'une boîte de taille
    exactement box_size x box_size, décalée si nécessaire pour rester
    entièrement dans l'image.

    Parameters
    ----------
    centroid : tuple(int, int)
        (row, col)
    box_size : int
        Taille de la boîte.
    img_shape : tuple(int, int)
        (height, width)
    """
    cy, cx = centroid
    half = box_size // 2

    # Lignes
    r_min = cy - half
    r_max = r_min + box_size

    if r_min < 0:
        r_min = 0
        r_max = box_size
    elif r_max > img_shape[0]:
        r_max = img_shape[0]
        r_min = r_max - box_size

    # Colonnes
    c_min = cx - half
    c_max = c_min + box_size

    if c_min < 0:
        c_min = 0
        c_max = box_size
    elif c_max > img_shape[1]:
        c_max = img_shape[1]
        c_min = c_max - box_size

    return r_min, r_max, c_min, c_max


def get_3d_centroid(seg: np.ndarray) -> tuple[int, int] | None:
    """
    Retourne (row_center, col_center) du centroïde 3D de la segmentation,
    projeté sur les axes (rows, cols) — indépendant de l'axe slice.
    """
    nonzero = np.argwhere(seg != 0)
    if len(nonzero) == 0:
        return None
    centroid = nonzero.mean(axis=0)  # [slice, row, col]
    return int(round(centroid[0])), int(round(centroid[1])), int(round(centroid[2]))


def process_case(case: str,
                 input_folder: str,
                 input_labels_folder: str,
                 input_labels_2_folder: str,
                 save_input_folder: str,
                 save_labels_folder: str,
                 save_labels_2_folder: str,
                 box_size: int = 256) -> str:

    file_nii = sitk.ReadImage(os.path.join(input_folder, case + "_0000.nii.gz"))
    file_labels_nii = sitk.ReadImage(os.path.join(input_labels_folder, case))
    file_labels_2_nii = sitk.ReadImage(os.path.join(input_labels_2_folder, case))
    spacing = file_labels_nii.GetSpacing()[::-1]

    images = sitk.GetArrayFromImage(file_nii)
    labels = sitk.GetArrayFromImage(file_labels_nii)
    labels_2 = sitk.GetArrayFromImage(file_labels_2_nii)


    # -------------------------
    # 1. RESIZE
    # -------------------------
    new_spacing = [spacing[0], spacing[1], 0.9722222089767456, 0.9722222089767456]
    new_shape = calculate_new_shape(spacing, labels.shape, new_spacing)

    print(case, spacing, labels.shape, new_spacing, new_shape)

    shape_smaller_than_box_size = False
    if new_shape[-2] < box_size:
        shape_smaller_than_box_size = True
        new_shape[-2] = box_size

    if new_shape[-1] < box_size:
        shape_smaller_than_box_size = True
        new_shape[-1] = box_size

    if shape_smaller_than_box_size:
        new_spacing = (
                np.array(spacing) * np.array(labels.shape) / np.array(new_shape)
        ).tolist()
        print("Shape adapted, new spacing is: ", spacing)

    images = resize(images, new_shape, order=3, mode="edge", cval=0,
                    clip=True, anti_aliasing=False)
    labels = resize(labels, new_shape, order=0, mode="edge", cval=0,
                    preserve_range=True, anti_aliasing=False).astype(labels.dtype)
    labels_2 = resize(labels_2, new_shape, order=0, mode="edge", cval=0,
                      preserve_range=True, anti_aliasing=False).astype(labels_2.dtype)

    # -------------------------
    # 2. ROI_BOX
    # -------------------------
    centroid_3d = get_3d_centroid(labels[0])

    if centroid_3d is not None:
        r_min_3d, r_max_3d, c_min_3d, c_max_3d = get_roi_box(
            centroid_3d[1:], box_size, labels[0].shape[1:]
        )
        images = images[:, :, r_min_3d:r_max_3d, c_min_3d:c_max_3d]
        labels = labels[:, :, r_min_3d:r_max_3d, c_min_3d:c_max_3d]
        labels_2 = labels_2[:, :, r_min_3d:r_max_3d, c_min_3d:c_max_3d]

    # -------------------------
    # 3. INTENSITY
    # -------------------------

    images = scale_intensity(images)

    # -------------------------
    # 4. SAVE NIFTI
    # -------------------------
    # ascontiguousarray : le crop ci-dessus produit des vues non
    # contiguës ; SimpleITK doit sinon les recopier lui-même en
    # interne, ce qu'on préfère faire explicitement en numpy (plus
    # rapide, et rend l'intention claire).
    img_sitk = sitk.GetImageFromArray(np.ascontiguousarray(images), isVector=False)
    lbl_sitk = sitk.GetImageFromArray(np.ascontiguousarray(labels), isVector=False)
    lbl_2_sitk = sitk.GetImageFromArray(np.ascontiguousarray(labels_2), isVector=False)

    img_sitk.SetSpacing(new_spacing[::-1])
    lbl_sitk.SetSpacing(new_spacing[::-1])
    lbl_2_sitk.SetSpacing(new_spacing[::-1])

    sitk.WriteImage(img_sitk, os.path.join(save_input_folder, case + "_0000.nii.gz"))
    sitk.WriteImage(lbl_sitk, os.path.join(save_labels_folder, case + ".nii.gz"))
    sitk.WriteImage(lbl_2_sitk, os.path.join(save_labels_2_folder, case + ".nii.gz"))

    return case


if __name__ == "__main__":
    box_size = 256
    dataset = "ALL"
    input_folder = f"C:/Users/goujat/Documents/thesis/LDMforCardiacMRI/data/{dataset}/PEPR_CINE_SA_ED/raw/Train_images4DTs"
    save_input_folder = f"C:/Users/goujat/Documents/thesis/LDMforCardiacMRI/data/{dataset}/PEPR_CINE_SA_ED/raw/Train_images4DTsProcess"

    input_labels_folder = f"C:/Users/goujat/Documents/thesis/LDMforCardiacMRI/data/{dataset}/PEPR_CINE_SA_ED/raw/Train_mean_probamap4D_APHM"
    save_labels_folder = f"C:/Users/goujat/Documents/thesis/LDMforCardiacMRI/data/{dataset}/PEPR_CINE_SA_ED/raw/Train_mean_probamap4D_APHM_Process"

    input_labels_2_folder = f"C:/Users/goujat/Documents/thesis/LDMforCardiacMRI/data/{dataset}/PEPR_CINE_SA_ED_ES/raw/Train_mean_probamap4D_APHM"
    save_labels_2_folder = f"C:/Users/goujat/Documents/thesis/LDMforCardiacMRI/data/{dataset}/PEPR_CINE_SA_ED_ES/raw/Train_mean_probamap4D_APHM_Process"

    files = subfiles(input_labels_folder, suffix=".nii.gz", join=False, sort=True)
    case_ids = np.unique([i[:-7] for i in files])

    os.makedirs(save_input_folder, exist_ok=True)
    os.makedirs(save_labels_folder, exist_ok=True)
    os.makedirs(save_labels_2_folder, exist_ok=True)

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                process_case, case,
                input_folder, input_labels_folder, input_labels_2_folder,
                save_input_folder, save_labels_folder, save_labels_2_folder,
                box_size
            ): case
            for case in case_ids[675:]
        }

        for future in as_completed(futures):
            case = futures[future]
            future.result()
            print(f"OK   {case}")


