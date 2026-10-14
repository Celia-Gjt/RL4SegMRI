import random
from typing import Iterable, List
from scipy import ndimage
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure, distance_transform_cdt
from vital.data.pepr.config import Label
import numpy as np

STRUCTURE = generate_binary_structure(2, 2)


def sample_hole_radius(
        radius_range=(1.0, 5.0),
) -> float:
    """
    Échantillonne un rayon avec une forte probabilité pour les petits trous.
    """
    r_min, r_max = radius_range

    # Distribution biaisée vers les petites valeurs
    u = np.random.beta(2.0, 5.0)

    return r_min + u * (r_max - r_min)


def get_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Retourne le contour interne d'un masque.
    """
    eroded = ndimage.binary_erosion(
        mask,
        structure=STRUCTURE,
        iterations=1,
    )

    return mask & ~eroded


def get_safe_interior(
        mask: np.ndarray,
        min_distance: int = 2,
) -> np.ndarray:
    """
    Retourne les pixels suffisamment éloignés du contour.

    Permet d'éviter de créer accidentellement des trous qui touchent
    immédiatement la frontière de la structure.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    distance = ndimage.distance_transform_edt(mask)

    return mask & (distance >= min_distance)


def create_irregular_hole(
        shape: tuple,
        center: tuple,
        radius: float,
) -> np.ndarray:
    """
    Crée un trou approximativement circulaire mais légèrement irrégulier.

    La forme est obtenue à partir d'une dilatation morphologique puis
    légèrement perturbée pour éviter les cercles parfaits.
    """
    seed = np.zeros(shape, dtype=bool)
    seed[center] = True

    hole = ndimage.binary_dilation(
        seed,
        structure=STRUCTURE,
        iterations=max(1, int(round(radius))),
    )

    if random.random() < 0.7:
        noise = np.random.random(shape)

        noise = ndimage.gaussian_filter(
            noise.astype(np.float32),
            sigma=1.0,
        )

        threshold = np.quantile(noise[hole], 0.35)

        irregularity = hole & (noise >= threshold)

        if irregularity.sum() > 0:
            hole = irregularity

    return hole


def punch_hole_in_structure(
        action: np.ndarray,
        target_label: int,
        hole_radius_range=(1, 5),
        num_holes: int = 1,
        min_boundary_distance: int = 2,
) -> np.ndarray:
    """
    Crée des défauts locaux irréguliers à l'intérieur d'une structure.

    Les trous :
      - sont préférentiellement petits ;
      - sont placés dans l'intérieur de la structure ;
      - ne débordent pas sur les structures voisines ;
      - ont une forme légèrement irrégulière ;
      - peuvent être espacés pour éviter de créer un gros trou artificiel.
    """

    degraded = action.copy()

    structure_mask = action == target_label

    if not np.any(structure_mask):
        return degraded

    safe_region = get_safe_interior(
        structure_mask,
        min_distance=min_boundary_distance,
    )

    if not np.any(safe_region):
        safe_region = structure_mask

    candidate_points = np.argwhere(safe_region)

    if len(candidate_points) == 0:
        return degraded

    occupied_holes = np.zeros_like(structure_mask, dtype=bool)

    for _ in range(num_holes):

        if len(candidate_points) == 0:
            break

        idx = np.random.randint(len(candidate_points))
        center = tuple(candidate_points[idx])

        radius = sample_hole_radius(
            hole_radius_range
        )

        hole = create_irregular_hole(
            structure_mask.shape,
            center,
            radius,
        )

        hole &= structure_mask
        hole &= ~occupied_holes

        if not np.any(hole):
            continue

        degraded[hole] = Label.BG

        occupied_holes |= hole

        candidate_region = safe_region & ~occupied_holes

        candidate_points = np.argwhere(candidate_region)

    return degraded


def punch_hole_at_interface(
        action: np.ndarray,
        label_a: int,
        label_b: int,
        hole_radius_range=(1, 3),
        num_holes: int = 1,
) -> np.ndarray:
    """
    Crée des petits défauts locaux au niveau de l'interface entre
    deux structures.

    Le défaut peut supprimer une petite partie de A, de B, ou des deux.
    Il reste strictement limité à l'interface.
    """

    degraded = action.copy()

    mask_a = action == label_a
    mask_b = action == label_b

    if not np.any(mask_a) or not np.any(mask_b):
        return degraded

    interface_a = (
            mask_a
            & ndimage.binary_dilation(
        mask_b,
        structure=STRUCTURE,
        iterations=1,
    )
    )

    interface_b = (
            mask_b
            & ndimage.binary_dilation(
        mask_a,
        structure=STRUCTURE,
        iterations=1,
    )
    )

    interface = interface_a | interface_b

    if not np.any(interface):
        return degraded

    candidate_points = np.argwhere(interface)

    for _ in range(num_holes):

        if len(candidate_points) == 0:
            break

        idx = np.random.randint(len(candidate_points))
        center = tuple(candidate_points[idx])

        radius = sample_hole_radius(
            hole_radius_range
        )

        hole = create_irregular_hole(
            interface.shape,
            center,
            radius,
        )

        hole &= (mask_a | mask_b)

        if not np.any(hole):
            continue

        defect_type = random.choices(
            [
                "remove_a",
                "remove_b",
                "remove_both",
            ],
            weights=[0.35, 0.35, 0.30],
            k=1,
        )[0]

        if defect_type == "remove_a":

            degraded[hole & mask_a] = Label.BG

        elif defect_type == "remove_b":

            degraded[hole & mask_b] = Label.BG

        else:

            degraded[hole & (mask_a | mask_b)] = Label.BG

    return degraded


def remove_rv(
        action: np.ndarray,
        replacement_label: int = Label.BG,
        fraction_range=(0.3, 1.0),
) -> np.ndarray:
    """
    Supprime une partie ou la totalité du RV.
    """

    degraded = action.copy()

    rv = action == Label.RV

    if not np.any(rv):
        return degraded

    fraction = random.uniform(
        *fraction_range
    )

    # Suppression quasi complète
    if fraction >= 0.95:
        degraded[rv] = replacement_label
        return degraded

    # Choisir une zone locale du RV
    region = get_random_contour_region(
        rv,
        min_width=0.2,
        max_width=0.5,
    )

    degraded[region] = replacement_label

    return degraded


def sample_deformation_size() -> int:
    """
    Échantillonne une amplitude de déformation.

    La majorité des déformations sont petites, avec quelques déformations
    plus importantes.
    """
    return random.choices(
        population=[1, 2, 3, 4],
        weights=[0.30, 0.40, 0.20, 0.10],
        k=1,
    )[0]


def get_random_contour_region(
        mask: np.ndarray,
        min_width: float = 0.20,
        max_width: float = 0.45,
) -> np.ndarray:
    """
    Sélectionne une portion aléatoire du masque en fonction de l'angle
    autour de son centre de masse.

    Permet de faire une déformation locale plutôt que de modifier
    toute la structure.
    """
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return np.zeros_like(mask, dtype=bool)

    cx = xs.mean()
    cy = ys.mean()

    yy, xx = np.indices(mask.shape)

    angles = np.arctan2(
        yy - cy,
        xx - cx,
    )

    center_angle = random.uniform(-np.pi, np.pi)

    width = random.uniform(
        min_width * np.pi,
        max_width * np.pi,
    )

    angular_difference = np.angle(
        np.exp(1j * (angles - center_angle))
    )

    region = np.abs(angular_difference) < width / 2

    return region & mask


"""def erode_structure(
        action: np.ndarray,
        target_label: int,
        replacement_label: int,
        n_iter: int = 1,
        local: bool = True,
) -> np.ndarray:

    degraded = action.copy()

    mask = action == target_label

    if not np.any(mask):
        return degraded

    if local:
        region = get_random_contour_region(mask)
        if not np.any(region):
            return degraded

        eroded_region = binary_erosion(
            region,
            structure=STRUCTURE,
            iterations=n_iter,
        )

        removed = region & ~eroded_region
    else:
        eroded_mask = binary_erosion(
            mask,
            structure=STRUCTURE,
            iterations=n_iter,
        )

        removed = mask & ~eroded_mask

    degraded[removed] = replacement_label

    return degraded """

def erode_structure(
        action: np.ndarray,
        target_label: int,
        replacement_label: int,
        n_iter: int = 1,
        local: bool = True,
        contact_label: int | List[int] = Label.BG,
) -> np.ndarray:
    """
    Érode une structure uniquement du côté qui touche la ou les structures contact sélectionnées.
    Les interfaces avec les autres structures ne sont pas touchées.
    """
    degraded = action.copy()

    mask = action == target_label
    if not np.any(mask):
        return degraded

    contact_labels = np.atleast_1d(contact_label)  # 0 -> [0], [0, 4] -> [0, 4]
    is_contact = np.isin(action, contact_labels)

    dist_to_bg = distance_transform_cdt(~is_contact, metric="chessboard")

    # pixels de la structure à moins de n_iter pixels du background
    removed = mask & (dist_to_bg <= n_iter)

    if local:
        region = get_random_contour_region(mask)
        if not np.any(region):
            return degraded
        removed &= region

    if not np.any(removed):
        return degraded

    degraded[removed] = replacement_label
    return degraded


def dilate_structure(
        action: np.ndarray,
        target_label: int,
        allowed_labels: Iterable[int],
        n_iter: int = 1,
        local: bool = True,
) -> np.ndarray:
    """
    Dilate localement ou globalement une structure.

    La structure peut uniquement empiéter sur les labels spécifiés dans
    `allowed_labels`.

    Cela évite par exemple qu'une dilation du LV écrase arbitrairement
    une structure anatomiquement incompatible.
    """
    degraded = action.copy()

    mask = action == target_label

    if not np.any(mask):
        return degraded

    allowed_labels = np.asarray(list(allowed_labels))

    if local:
        region = get_random_contour_region(mask)

        if not np.any(region):
            return degraded

        dilated_region = binary_dilation(
            region,
            structure=STRUCTURE,
            iterations=n_iter,
        )
    else:
        dilated_region = binary_dilation(
            mask,
            structure=STRUCTURE,
            iterations=n_iter,
        )

    gained = (
            dilated_region
            & ~mask
            & np.isin(action, allowed_labels)
    )

    degraded[gained] = target_label

    return degraded


def shift_interface(
        action: np.ndarray,
        label_a: int,
        label_b: int,
        n_iter: int = 1,
) -> np.ndarray:
    """
    Déplace localement l'interface entre deux structures.

    Seuls les pixels appartenant à l'interface A/B peuvent être modifiés.
    Le fond et les autres structures sont conservés.
    """
    degraded = action.copy()

    mask_a = action == label_a
    mask_b = action == label_b

    if not np.any(mask_a) or not np.any(mask_b):
        return degraded

    interface_a = (
            mask_a
            & binary_dilation(
        mask_b,
        structure=STRUCTURE,
        iterations=1,
    )
    )

    interface_b = (
            mask_b
            & binary_dilation(
        mask_a,
        structure=STRUCTURE,
        iterations=1,
    )
    )

    if not np.any(interface_a) or not np.any(interface_b):
        return degraded

    if random.random() < 0.5:
        source = interface_a

        region = get_random_contour_region(source)

        if not np.any(region):
            return degraded

        expansion = binary_dilation(
            region,
            structure=STRUCTURE,
            iterations=n_iter,
        )

        gained = expansion & mask_b

        degraded[gained] = label_a

    else:
        source = interface_b

        region = get_random_contour_region(source)

        if not np.any(region):
            return degraded

        expansion = binary_dilation(
            region,
            structure=STRUCTURE,
            iterations=n_iter,
        )

        gained = expansion & mask_a

        degraded[gained] = label_b

    return degraded
