import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage import measure, morphology
from skimage.morphology import convex_hull_image
from vital.data.pepr.config import Label

from rl4seg.datamodules.PEPR_RL_datamodule import PEPRDataModule

thresholds = {
    'lv_concavity': {'max': 4},
    'rv_concavity': {'max': 30},
    'myo_concavity': {'max': 3},
    'lv_circularity': {'min': 0.2},
    'myo_circularity': {'min': 0.2},
}

thresholds_base = {
    'lv_concavity': {'max': 20},
    'rv_concavity': {'max': 30},
    'myo_concavity': {'max': 20},
    'lv_circularity': {'min': 0.1},
    'myo_circularity': {'min': 0.1},
}

# --- Trous internes à un label ---
INTERNAL_HOLE_METRIC_TO_LABEL = {
    'holes_in_lv': Label.LV,
    'holes_in_rv': Label.RV,
    'holes_in_myo': Label.MYO,
}

# --- Trous "coincés" entre deux labels adjacents ---
FRONTIER_HOLE_METRIC_TO_LABELS = {
    'holes_between_lv_and_myo': (Label.LV, Label.MYO),
    'holes_between_rv_and_myo': (Label.RV, Label.MYO),
}

LV_NON_ENCLOSURE_THRESHOLD = 1 / 3


def _fill_label_holes(segmentation, label_value):
    """ Rebouche les trous internes d'un label donné dans la segmentation.

    Args:
        segmentation: ndarray 2D, la segmentation d'origine.
        label_value: int, la valeur du label dont on veut combler les trous.

    Returns:
        ndarray, la segmentation avec les trous du label comblés.
    """
    mask = (segmentation == label_value)
    filled_mask = binary_fill_holes(mask)

    corrected = segmentation.copy()
    newly_filled = filled_mask & ~mask
    corrected[newly_filled] = label_value

    return corrected


def _fill_holes_between_labels(segmentation, labels_involved, background_label):
    """ Rebouche les pixels de background piégés entre deux labels adjacents,
    en attribuant à chaque pixel le label du pixel non-background le plus proche
    (parmi les labels concernés).

    Args:
        segmentation: ndarray 2D, la segmentation d'origine.
        labels_involved: tuple(int, int), les deux labels entre lesquels
                          des trous de background peuvent apparaître (ex: (lv, myo)).
        background_label: int, la valeur du label "background".

    Returns:
        ndarray, la segmentation avec les pixels piégés réattribués.
    """
    labels_involved = list(labels_involved)

    # Masque de l'union des deux structures concernées.
    union_mask = np.isin(segmentation, labels_involved)

    # En rebouchant les trous de cette union, on récupère aussi les pixels
    # de background qui étaient "enfermés" entre les deux labels.
    filled_union = binary_fill_holes(union_mask)

    # On ne corrige que les pixels réellement labellisés "background" à l'origine,
    # pour ne jamais écraser une autre structure (ex: rv qui se serait glissée là).
    hole_mask = filled_union & ~union_mask & (segmentation == background_label)

    if not np.any(hole_mask):
        return segmentation.copy()

    # Pour chaque pixel du trou, on cherche le pixel labellisé (lv ou myo, ou rv ou myo)
    # le plus proche, et on lui attribue son label -> "vote" par proximité spatiale.
    indices = distance_transform_edt(
        ~union_mask, return_distances=False, return_indices=True
    )
    nearest_labels = segmentation[tuple(indices)]

    corrected = segmentation.copy()
    corrected[hole_mask] = nearest_labels[hole_mask]

    return corrected


def correct_holes_in_segmentation(segmentation, metrics, background_label=Label.BG):
    """ Corrige les trous détectés dans une segmentation.
    Args:
        segmentation: ndarray 2D, la segmentation d'origine.
        metrics: dict, les métriques anatomiques calculées pour cette segmentation.
        background_label: int, la valeur du label "background" (par défaut `background`).

    Returns:
        tuple(ndarray, bool):
            - la segmentation corrigée (ou identique si aucun trou détecté),
            - un booléen indiquant si une correction a été appliquée.
    """
    corrected = segmentation.copy()
    correction_applied = False

    for metric_name, label_value in INTERNAL_HOLE_METRIC_TO_LABEL.items():
        if metrics.get(metric_name):
            corrected = _fill_label_holes(corrected, label_value)
            correction_applied = True
            print(f"Correction appliquée : {metric_name} (label={label_value})")

    for metric_name, labels_involved in FRONTIER_HOLE_METRIC_TO_LABELS.items():
        if metrics.get(metric_name):
            corrected = _fill_holes_between_labels(corrected, labels_involved, background_label)
            correction_applied = True
            print(f"Correction appliquée : {metric_name} (labels={labels_involved})")

    return corrected, correction_applied


def check_segmentation_validity(segmentation, voxelspacing, is_base, **kwargs):
    """ Wrapper around the function that computes anatomical metrics for segmentations that is only concerned with
    whether any anatomical errors where detected in the segmentation or not.

    Args:
        segmentation: ndarray, segmentation for which to compute anatomical metrics.
        voxelspacing: tuple, the size of the segmentation's voxels along each (height, width) dimension (in mm).
        is_base: Booleen, True if slice is in base or apex.
        **kwargs: additional parameters.

    Returns:
        bool, whether the segmentation is anatomically plausible (True) or not (False).
    """
    metrics = _extract_anatomical_metrics_by_segmentation(segmentation, voxelspacing, is_base)
    return not metrics['anatomical_errors'], metrics


def check_metric_validity(metric_value: float, thresholds: dict, optional_structure: bool) -> bool:
    """ Checks whether the value of the metric is within the range of values meaning the segmentation is correct.

    Args:
        metric_value: float, value of the metric for a segmentation.
        thresholds: dict, minimum ('min') and maximum ('max') tolerated value for the metric. Any value below the
                    minimum or above the maximum will lead to the segmentation being considered erroneous.
                    If no thresholds are specified, a value of '0' is considered to mean no error, and any other value
                    means an error.
        optional_structure: bool, indicate whether the segmentation can be considered valid even if the metric' value
                            indicates the structure for which it was measured is absent.

    Returns:
        bool, whether the value of the metric is within the range of values meaning the segmentation is correct.
    """
    if np.isnan(metric_value):  # If the structure on which to compute the metric was not present in the segmentation
        validity = optional_structure
    elif thresholds:  # If the metric has a range of value within which segmentations are considered valid
        validity = thresholds.get('min', 0) <= metric_value <= thresholds.get('max', np.inf)
    else:  # If no range of valid values are provided, a value of '0' is considered to mean no error, and any other
        # value means an error
        validity = not bool(metric_value)
    return validity


def _extract_anatomical_metrics_by_segmentation(segmentation, voxelspacing, is_base):
    """ Computes the anatomical metrics for a single segmentation.

    Args:
        segmentation: ndarray, the segmentation as a 2d array.
        voxelspacing: tuple, the size of the segmentation's voxels along each (height, width) dimension (in mm).
        is_base: Booleen, True if slice is in base.

    Returns:
        dict, mapping between the anatomical metrics' names and their value for the segmentation.
    """
    segmentation_metrics = Segmentation2DMetrics(segmentation, [Label.BG, Label.LV, Label.MYO, Label.RV],
                                                 voxelspacing=voxelspacing)
    lv_metrics = LeftVentricleMetrics(segmentation_metrics)
    myo_metrics = MyocardiumMetrics(segmentation_metrics)
    rv_metrics = RightVentricleMetrics(segmentation_metrics)

    frontier_metrics = FrontierMetrics(segmentation_metrics)
    metrics = {
        'holes_in_lv': lv_metrics.count_holes(),
        'holes_in_rv': rv_metrics.count_holes(),
        'holes_in_myo': myo_metrics.count_holes(),
        'disconnectivity_in_lv': lv_metrics.count_disconnectivity(),
        'disconnectivity_in_rv': rv_metrics.count_disconnectivity(),
        'disconnectivity_in_myo': myo_metrics.count_disconnectivity(),
        'holes_between_lv_and_myo': frontier_metrics.holes_between_lv_and_myo(),
        # Il ne peut pas y avoir de trous entre LV et myo: devrait être valide
        'holes_between_rv_and_myo': frontier_metrics.holes_between_rv_and_myo(),
        # Il ne peut pas y avoir de trous entre RV et myo: devrait être valide
        'rv_disconnected_from_myo': frontier_metrics.rv_disconnected_from_myo(),  # RV déconnecté du myo: ?
        'frontier_between_lv_and_rv': frontier_metrics.frontier_between_lv_and_rv(),
        # Le lv ne peut pas être en contact avec rv: erreur à la base
        'frontier_between_lv_and_background': frontier_metrics.frontier_between_lv_and_background(),
        # Le lv ne peut pas être en contact avec background: erreur à la base
        'lv_concavity': lv_metrics.measure_concavity(),
        'rv_concavity': rv_metrics.measure_concavity(),
        'myo_concavity': myo_metrics.measure_concavity(),
        "lv_circularity": lv_metrics.measure_circularity(),
        "myo_circularity": myo_metrics.measure_circularity(),
    }
    anatomical_errors = False
    for name, value in metrics.items():
        if is_base:
            if not check_metric_validity(value, thresholds_base.get(name), optional_structure=True):
                if ((name == 'holes_between_rv_and_myo'
                     or name == 'frontier_between_lv_and_rv'
                     or name == 'frontier_between_lv_and_background')
                        and segmentation_metrics.measure_non_enclosure_ratio(struct_label=Label.LV,
                                                                             blocking_label=Label.MYO,
                                                                             other_labels=[Label.BG,
                                                                                           Label.RV]) < LV_NON_ENCLOSURE_THRESHOLD):
                    continue
                elif ((name == 'myo_concavity' or name == 'myo_circularity' or name == 'lv_concavity')
                      and ((not check_metric_validity(metrics['frontier_between_lv_and_background'],
                                                      thresholds_base.get('frontier_between_lv_and_background'),
                                                      optional_structure=True))
                           or (not check_metric_validity(metrics['frontier_between_lv_and_rv'],
                                                         thresholds_base.get('frontier_between_lv_and_rv'),
                                                         optional_structure=True)))):

                    continue

                anatomical_errors = True
                print("Error ", name)
        else:
            if not check_metric_validity(value, thresholds.get(name), optional_structure=True):
                anatomical_errors = True
                print("Error ", name)

    metrics['anatomical_errors'] = anatomical_errors

    return metrics


class Segmentation2DMetrics:
    """ Class that implements algorithms to compute metrics on segmentations.
    """

    def __init__(self, segmentation: np.ndarray, struct_labels: Union[int, list], voxelspacing: tuple = (1, 1)):
        """
        Args:
            segmentation: ndarray, a 2D array of dimensions (height, width) where the value of each entry in the array
                          is the label of the segmentation class for the pixel.
            struct_labels: int or list, the label(s) of the class(es) present in the segmentation for which to compute
                           metrics.
            voxelspacing: tuple. the size of the voxels along each (height, width) dimension (in mm).
        """
        try:
            iter(struct_labels)
        except TypeError:
            struct_labels = [struct_labels]  # Make struct_labels into an iterable if it isn't already

        self.segmentation = segmentation
        self.voxelspacing = voxelspacing
        self.binary_structs = {struct_label: np.isin(segmentation, struct_label).astype(dtype=np.uint8)
                               for struct_label in struct_labels}
        self.binary_structs_inverse = {struct_label: 1 - binary_struct
                                       for struct_label, binary_struct in self.binary_structs.items()}

        # Approximate size of small blobs determined through empirical experiments. Those small blobs correspond to
        # small artifacts left behind by some morphological operations on the images, and which can be ignored since
        # they would otherwise skew the metrics' values.
        self.small_objects_size = segmentation.shape[0] * segmentation.shape[1] / 2 ** 14

    def measure_non_enclosure_ratio(self, struct_label: Union[int, list], blocking_label: Union[int, list],
                                    other_labels: Union[int, list], n_angles: int = 360,
                                    max_radius: int = None) -> float:
        """ Mesure la proportion de directions (angles), depuis le centroïde de `struct_label`, pour lesquelles
        la première structure rencontrée en sortant de `struct_label` fait partie de `other_labels` plutôt que
        de `blocking_label` (structure censée entourer `struct_label`).

        Contrairement à un comptage de pixels de contour, cette méthode est robuste aux concavités de la forme :
        une encoche qui expose plusieurs pixels de bordure dans la même direction ne compte que pour UNE seule
        direction "ouverte", pas plusieurs.

        Args:
            struct_label: int ou list, label(s) de la structure dont on mesure l'exposition (ex: lv).
            blocking_label: int ou list, label(s) censé(s) entourer struct_label (ex: myo).
            other_labels: int ou list, label(s) dont le contact direct est anormal (ex: [rv, background]).
            n_angles: int, nombre de directions échantillonnées autour du centroïde.
            max_radius: int, distance max (en pixels) à parcourir le long du rayon. Par défaut, la diagonale
                        de l'image.

        Returns:
            float, ratio (0 à 1) de directions où struct_label est directement exposé à other_labels.
        """
        binary_struct = self.binary_structs[struct_label]
        if not np.amax(binary_struct):
            return 0.

        if max_radius is None:
            max_radius = int(np.hypot(*self.segmentation.shape)) + 1

        ys, xs = np.nonzero(binary_struct)
        cy, cx = ys.mean(), xs.mean()

        blocking_mask = np.isin(self.segmentation, blocking_label)
        other_mask = np.isin(self.segmentation, other_labels)
        binary_struct_bool = binary_struct.astype(bool)

        h, w = self.segmentation.shape
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        n_open = 0
        n_valid_angles = 0

        for theta in angles:
            dy, dx = np.sin(theta), np.cos(theta)
            for r in range(1, max_radius):
                y = int(round(cy + r * dy))
                x = int(round(cx + r * dx))
                if not (0 <= y < h and 0 <= x < w):
                    break  # sorti de l'image sans quitter struct_label (ne devrait pas arriver normalement)
                if binary_struct_bool[y, x]:
                    continue  # encore dans la structure elle-même, on continue d'avancer
                # premier pixel hors de struct_label rencontré le long du rayon
                n_valid_angles += 1
                if other_mask[y, x] and not blocking_mask[y, x]:
                    n_open += 1
                break
            # si aucun pixel hors struct_label n'est atteint (rayon trop court ou sorti de l'image),
            # cette direction est ignorée

        if n_valid_angles == 0:
            return 0.
        return n_open / n_valid_angles

    def count_holes(self, struct_label: Union[int, list]) -> int:
        """ Counts the pixels that form holes in a supposedly contiguous segmented area.

        Args:
            struct_label: int or list, the label(s) of the class(es) making up the segmented area for which to compute
                          the metric.

        Returns:
            int, number of pixels that form holes in the segmented area.
        """
        # Mark the class of interest as 0 and everything else as 1
        # Merge the regions of 1 that are open by a side using padding (these are not holes)
        binary_struct = self.binary_structs_inverse[struct_label]
        binary_struct = np.pad(binary_struct, ((1, 1), (1, 1)), 'constant', constant_values=1)

        # Extract properties of continuous regions of 1
        props = measure.regionprops(measure.label(binary_struct, connectivity=2))

        hole_pixel_count = 0
        for prop in props:
            # Skip the region open by the side (the one that includes padding)
            if prop.bbox[0] != 0:
                hole_pixel_count += prop.area

        return hole_pixel_count

    def count_disconnectivity(self, struct_label: Union[int, list]) -> int:
        """ Counts the pixels that are disconnected from a supposedly contiguous segmented area.

        Args:
            struct_label: int or list, the label(s) of the class(es) making up the segmented area for which to compute
                          the metric.

        Returns:
            int, total number of disconnected pixels in the segmented area.
        """
        binary_struct = self.binary_structs[struct_label]

        # Extract properties for every disconnected region of the supposedly continuous segmented area
        labels_props = measure.regionprops(measure.label(binary_struct, connectivity=2))

        if len(labels_props) > 1:  # If there is disconnectivity in the segmentated area
            # Sort by decreasing pixel count of the region
            labels_props_by_desc_size = sorted(labels_props, key=lambda k: k.area, reverse=True)

            total_minus_biggest = sum(label_props.area for label_props in labels_props_by_desc_size[1:])
            return total_minus_biggest

        else:  # If there was only a single contiguous region making up the segmented area
            return 0

    def count_holes_between_regions(self, struct1_label: Union[int, list], struct2_label: Union[int, list]) -> int:
        """ Counts the pixels in the gap between two supposedly connected segmented areas.

        NOTE: As this method iterates over holes (directly in Python) in supposedly filled-in segmented areas, its
        runtime will increase by multiple orders of magnitude if tested on degenerated results (e.g. similar to noise),
        where segmented areas are not filled-in and can be seen as full of a multitude of tiny holes.

        Args:
            struct1_label: int or list, the label(s) of the class(es) making up one of the segmented areas.
            struct2_label: int or list, the label(s) of the class(es) making up the other segmented area.

        Returns:
            int, number of pixels in the gap between the two segmented areas.
        """
        # Obtain binary segmentations for each segmented areas
        binary_struct1 = self.binary_structs[struct1_label]
        binary_struct2 = self.binary_structs[struct2_label]
        not_binary_struct1 = self.binary_structs_inverse[struct1_label]
        not_binary_struct2 = self.binary_structs_inverse[struct2_label]

        # Find holes inside the first segmented area
        holes_struct1, last_hole_label_class1 = measure.label(not_binary_struct1, connectivity=1, return_num=True)
        outside_class1 = holes_struct1[0, 0]
        holes_struct1[holes_struct1 == outside_class1] = 0

        # Find holes inside the second segmented area
        holes_class2, last_hole_label_class2 = measure.label(not_binary_struct2, connectivity=1, return_num=True)
        outside_class2 = holes_class2[0, 0]
        holes_class2[holes_class2 == outside_class2] = 0

        # Find the holes in the first area that don't contain the second area (and the opposite)
        hole_labels_class1_without_class2 = [h for h in range(1, last_hole_label_class1 + 1) if
                                             h not in np.unique(holes_struct1 * binary_struct2)]
        hole_labels_class2_without_class1 = [h for h in range(1, last_hole_label_class2 + 1) if
                                             h not in np.unique(holes_class2 * binary_struct1)]
        holes_class1_without_class2 = np.isin(holes_struct1, hole_labels_class1_without_class2, assume_unique=True) \
            .astype(dtype=np.uint8)
        holes_class2_without_class1 = np.isin(holes_class2, hole_labels_class2_without_class1, assume_unique=True) \
            .astype(dtype=np.uint8)

        not_both = 1 - ((binary_struct1 + holes_class1_without_class2 +
                         binary_struct2 + holes_class2_without_class1) > 0)
        holes_between = measure.label(not_both, connectivity=1)
        outside = holes_between[0, 0]
        holes_between[holes_between == outside] = 0
        pixels_in_holes = np.sum(holes_between > 0)

        return pixels_in_holes if pixels_in_holes > 1 else 0

    def count_frontier_between_regions(self, struct1_label: Union[int, list], struct2_label: Union[int, list]) -> int:
        """ Counts the pixels touching between two supposedly disconnected segmented areas.

        Args:
            struct1_label: int or list, the label(s) of the class(es) making up one of the segmented areas.
            struct2_label: int or list, the label(s) of the class(es) making up the other segmented area.

        Returns:
            int, number of pixels on the frontier between the two segmented areas.
        """
        # Obtain binary segmentations for each segmented area
        binary_struct1 = self.binary_structs[struct1_label]
        binary_struct2 = self.binary_structs[struct2_label]
        struct2_dilated = morphology.dilation(binary_struct2, np.ones((3, 3)))
        frontier = binary_struct1 * struct2_dilated
        pixels_on_frontier = frontier.sum()
        return pixels_on_frontier if pixels_on_frontier > 1 else 0

    def measure_concavity(self, struct_label: Union[int, list], no_structure_flag: float = float('nan')) -> float:
        """ Measures the depth of a concavity in a supposedly convex segmented area.

        Args:
            struct_label: int or list, the label(s) of the class(es) making up the segmented area for which to compute
                          the metric.
            no_structure_flag: float, value to return when no structure was found in the segmentation.

        Returns:
             float, the depth (in mm) of a concavity in the segmented area.
        """
        # Use binary structure with holes filled in
        # Necessary for structures that are not supposed to filled (e.g. myocardium)
        binary_struct = binary_fill_holes(self.binary_structs[struct_label]).astype(dtype=np.uint8)

        if np.amax(binary_struct):  # If the structure is present in the image

            # Get convex hull of the segmented area
            convex_hull = morphology.convex_hull_image(binary_struct)

            # Compute a mask for the difference between the convex hull and the source image
            # The mask is eroded to account for the behavior of ```distance_transform_edt``` where a pixel
            # on the frontier with the background is at distance 1 from the background
            diff_convex_img = convex_hull - binary_struct
            diff_convex_img = morphology.binary_erosion(diff_convex_img, np.ones((3, 3)))

            # Compute distance from convex hull pixels to nearest background pixel
            convex_dist_to_background = distance_transform_edt(convex_hull, sampling=self.voxelspacing)

            # Apply the convex hull difference mask to the distances
            convex_dist_to_background *= diff_convex_img

            # Get the Hausdorff distance by finding the maximum distance in the remaining distances
            hausdroff_distance = np.max(convex_dist_to_background)
            return hausdroff_distance

        else:  # If the structure is not in the image
            return no_structure_flag

    def measure_circularity(self, struct_label: Union[int, list], no_structure_flag: float = float('nan')) -> float:
        """ Measures the isoperimetric ratio of a segmented area, assuming the area is contiguous.

        Args:
            struct_label: int or list, the label(s) of the class(es) making up the segmented area for which to compute
                          the metric.
            no_structure_flag: float, value to return when no structure was found in the segmentation.

        Returns:
            float, the isoperimetric ratio of the segmented area.
        """
        # Use binary structure with holes filled in
        # Necessary for structures that are not supposed to filled (e.g. myocardium)
        binary_struct = binary_fill_holes(self.binary_structs[struct_label]).astype(dtype=np.uint8)

        if np.amax(binary_struct):  # If the structure is present in the image
            # Compute the perimeter of the given class
            perimeter = measure.perimeter(binary_struct, neighbourhood=4)
            if perimeter < 1e-6:
                return no_structure_flag

            # Compute the area of the given class
            area = measure.moments(binary_struct)[0, 0]
            if area < 1e-6:
                return no_structure_flag

            # Compute the isoperimetric ratio ( https://en.wikipedia.org/wiki/Isoperimetric_inequality )
            ratio = (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0
            return ratio

        else:  # If the structure is not in the image
            return no_structure_flag

    def measure_frontier_ratio_between_regions(self, struct1_label: Union[int, list], struct2_label: Union[int, list]) \
            -> float:
        """ Measures the ratio between the largest continuous segment of the frontier between the two segmented areas
        and the width of the first segmented area.
        Note that because of this behavior, `struct1_label` and `struct2_label` are note interchangeable.

        Args:
            struct1_label: int or list, the label(s) of the class(es) making up one of the segmented areas.
            struct2_label: int or list, the label(s) of the class(es) making up the other segmented area.

        Returns:
            float, the ratio between the largest continuous segment of the frontier and the width of the first
            segmented area.
        """
        # Measure the number of pixels on the frontier
        pixels_on_frontier = self.count_frontier_between_regions(struct1_label, struct2_label)

        # Measure the length (in pixel) of the width of the first segmented area
        binary_struct1 = self.binary_structs[struct1_label]
        if np.amax(binary_struct1):  # If the structure is present in the image
            binary_struct1_bbox = measure.regionprops(binary_struct1)[0].bbox
            binary_struct1_width = binary_struct1_bbox[3] - binary_struct1_bbox[1]  # 1 = bbox min col, 3 = bbox max col

            # Find the ratio between the largest continuous segment of the frontier and the width of the first segmented
            # area
            frontier_proportion = pixels_on_frontier / binary_struct1_width
            return frontier_proportion

        else:  # If the structure is not in the image
            return 0.

    def measure_width_ratio_between_regions(self, struct1_label: Union[int, list], struct2_label: Union[int, list],
                                            no_structure_flag: float = float('nan')) -> float:
        """ Measures the ratio between the width (not necessarily contiguous) of two structures at the center of the
        image.

        Args:
            struct1_label: int or list, the label(s) of the class(es) making up one of the segmented areas.
            struct2_label: int or list, the label(s) of the class(es) making up the other segmented area.
            no_structure_flag: float, value to return if either of the structure is not found in the segmentation.

        Returns:
            float, ratio between the width (not necessarily contiguous) of two structures at the center of the image.
        """
        center_row_idx = self.segmentation.shape[0] // 2
        struct1_width = np.sum(self.binary_structs[struct1_label][center_row_idx])
        struct2_width = np.sum(self.binary_structs[struct2_label][center_row_idx])
        return struct1_width / struct2_width if (struct1_width and struct2_width) else no_structure_flag

    def measure_erosion_ratio_before_split(self, struct_label: Union[int, list],
                                           no_structure_flag: float = float('nan')) -> float:
        """ Measures the ratio between the depth of erosion necessary to divide a continuous structure in at least two
        fragments and the maximum thickness (in pixels) of the structure.

        Args:
            struct_label: int or list, the label(s) of the class(es) making up the segmented area for which to compute
                          the metric.
            no_structure_flag: float, value to return when no structure was found in the segmentation.

        Returns:
            float, ratio between the depth of erosion necessary to divide a continuous structure in at least two
            fragments and the maximum thickness (in pixels) of the structure.
        """
        binary_struct = self.binary_structs[struct_label]
        if np.amax(binary_struct):  # If the structure is present in the image

            # Compute maximum thickness of the structure (pixel with maximum distance to nearest background pixel)
            struct_dist_to_background = distance_transform_edt(binary_struct)
            max_struct_dist_to_background = np.max(struct_dist_to_background)

            # Erode the continuous structure until it splits in at least two fragments
            eroding_binary_struct = binary_struct
            nb_erosions = 0
            while measure.label(eroding_binary_struct, connectivity=2, return_num=True)[1] == 1:
                eroding_binary_struct = morphology.binary_erosion(eroding_binary_struct, np.ones((3, 3)))
                morphology.remove_small_objects(eroding_binary_struct, min_size=self.small_objects_size, in_place=True)
                nb_erosions += 1

            return nb_erosions / max_struct_dist_to_background

        else:  # If the structure is not in the image
            return no_structure_flag

    def measure_convexity(self, struct_label: Union[int, list], no_structure_flag: float = float('nan')) -> float:
        """ Measures the shape convexity of a segmented area.

        Args:
            struct_label: int or list, the label(s) of the class(es) making up the segmented area for which to compute
                          the metric.
            no_structure_flag: float, value to return when no structure was found in the segmentation.

        Returns:
            float, the value of the shape convexity metric for the segmented area.
        """
        binary_struct = self.binary_structs[struct_label]
        if np.amax(binary_struct):  # If the structure is present in the image
            struct_convex_hull = convex_hull_image(binary_struct)
            convexity = np.sum(binary_struct) / np.sum(struct_convex_hull)
            return convexity

        else:  # If the structure is not in the image
            return no_structure_flag


class Anatomical2DStructureMetrics:
    """ Abstract class that implements overhead code to compute metrics for a specific anatomical structure in a
    segmentation.
    """

    def __init__(self, segmentation_metrics: Segmentation2DMetrics, struct_label: Union[int, tuple]):
        """
        Args:
            segmentation_metrics: Segmentation2DMetrics, an instance, based on the segmentation for which to compute
                                  anatomical metrics, of the class implementing various segmentation metrics.
            struct_label: int or tuple, the label(s) of the class(es) making up the anatomical structure for which to
                          compute anatomical metrics.
        """
        self.segmentation_metrics = segmentation_metrics
        self.struct_label = struct_label
        self.no_structure_flag = float('nan')

    def count_holes(self):
        """ Counts the pixels that form holes in the segmentation of the anatomical structure.

        Returns:
            int, the count of pixels that form holes in the segmentation of the anatomical structure.
        """
        return self.segmentation_metrics.count_holes(self.struct_label)

    def count_disconnectivity(self):
        """ Counts the pixels that are disconnected from the main area segmented as the anatomical structure.

        Returns:
            int, the count of pixels that are disconnected from the main area segmented as the anatomical structure.
        """
        return self.segmentation_metrics.count_disconnectivity(self.struct_label)

    def measure_concavity(self):
        """ Measures the depth of a concavity in a supposedly convex anatomical structure.

        Returns:
             float, the depth (in mm) of a concavity in the anatomical structure.
        """
        return self.segmentation_metrics.measure_concavity(self.struct_label,
                                                           no_structure_flag=self.no_structure_flag)

    def measure_circularity(self):
        """ Measures the isoperimetric ratio for an anatomical structure assuming the structure is contiguous.

        Returns:
            float, the isoperimetric ratio of the anatomical structure.
        """
        return self.segmentation_metrics.measure_circularity(self.struct_label,
                                                             no_structure_flag=self.no_structure_flag)

    def measure_erosion_ratio_before_split(self):
        """ Measures the ratio between the depth of erosion necessary to divide a continuous anatomical structure in at
        least two fragments and the maximum thickness (in pixels) of the structure.

        Returns:
            float, the ratio between the depth of erosion necessary to divide a continuous anatomical structure in at
            least two fragments and the maximum thickness (in pixels) of the structure.
        """
        return self.segmentation_metrics.measure_erosion_ratio_before_split(self.struct_label,
                                                                            no_structure_flag=self.no_structure_flag)

    def measure_convexity(self):
        """  Measures the shape convexity of the anatomical structure's segmentation.

        Returns:
            float, the value of the shape convexity metric for the anatomical structure's segmentation.
        """
        return self.segmentation_metrics.measure_convexity(self.struct_label,
                                                           no_structure_flag=self.no_structure_flag)


class FrontierMetrics:
    """ Class to compute metrics on the frontiers between segmentations of multiple anatomical structures.
    """

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """
        Args:
            segmentation_metrics: Segmentation2DMetrics, an instance, based on the segmentation for which to compute
                                  anatomical metrics, of the class implementing various segmentation metrics.
        """
        self.segmentation_metrics = segmentation_metrics

    def holes_between_lv_and_myo(self):
        """ Counts the pixels in the gap between the two areas segmented as LV and MYO.

        Returns:
             int, the count of pixels in the gap between the two areas segmented as LV and MYO.
        """
        return self.segmentation_metrics.count_holes_between_regions(Label.LV, Label.MYO)

    def holes_between_rv_and_myo(self):
        """ Counts the pixels in the gap between the two areas segmented as RV and MYO.

        Returns:
             int, the count of pixels in the gap between the two areas segmented as RV and MYO.
        """
        return self.segmentation_metrics.count_holes_between_regions(Label.RV, Label.MYO)

    def rv_disconnected_from_myo(self):
        """ Measures the gap (in mm) between the areas segmented as RV and MYO.

        Returns:
             float, the width of the gap (in mm) between the areas segmented as RV and MYO.
        """
        infinity = 1.5 * np.prod(self.segmentation_metrics.segmentation.shape) \
                   * max(
            self.segmentation_metrics.voxelspacing)  # this should be bigger than any distance between 2 pixels!
        rv_as_1 = morphology.dilation(np.isin(self.segmentation_metrics.segmentation, Label.RV).astype(dtype=np.uint8),
                                      np.ones((3, 3)))

        myo_as_0 = np.isin(self.segmentation_metrics.segmentation, Label.MYO, invert=True).astype(dtype=np.uint8)
        if rv_as_1.sum() > 0 and (1 - myo_as_0).sum() > 0:  # If both rv and myo are present in the image
            distance_to_myo = distance_transform_edt(myo_as_0, self.segmentation_metrics.voxelspacing)
            min_distance_rv_to_myo = (rv_as_1 * distance_to_myo + infinity * (1 - rv_as_1)).min()
            return min_distance_rv_to_myo
        else:  # If the image has no rv and myo
            return 0

    def frontier_between_lv_and_rv(self):
        """ Counts the pixels that touch between the areas segmented as LV and RV.

        Returns:
             int, the count of pixels that touch between the areas segmented as LV and RV.
        """
        return self.segmentation_metrics.count_frontier_between_regions(Label.LV, Label.RV)

    def frontier_between_lv_and_background(self):
        """ Counts the pixels that touch between the areas segmented as LV and background.

        Returns:
             int, the count of pixels that touch between the areas segmented as LV and background.
        """
        return self.segmentation_metrics.count_frontier_between_regions(Label.LV, Label.BG)


class LeftVentricleMetrics(Anatomical2DStructureMetrics):
    """ Class to compute metrics on the segmentation of the left ventricle.
    """

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """
        Args:
            segmentation_metrics: Segmentation2DMetrics, an instance, based on the segmentation for which to compute
                                  anatomical metrics, of the class implementing various segmentation metrics.
        """
        super().__init__(segmentation_metrics, Label.LV)


class MyocardiumMetrics(Anatomical2DStructureMetrics):
    """ Class to compute metrics on the segmentation of the myocardium.
    """

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """
        Args:
            segmentation_metrics: Segmentation2DMetrics, an instance, based on the segmentation for which to compute
                                  anatomical metrics, of the class implementing various segmentation metrics.
        """
        super().__init__(segmentation_metrics, Label.MYO)

    def count_holes(self):
        """ Counts the pixels that form holes in the supposedly contiguous area segmented as myocardium.

        Returns:
            int, number of pixels that form holes in the segmented area as myocardium.
        """
        holes_on_myo_count = super().count_holes()

        if holes_on_myo_count > 0:  # If myo has holes, assume it is closed and account for lv amongst the holes

            # Count the number of pixels that make up the lv segmented area (that should form a hole inside myo)
            lv_binary_struct = self.segmentation_metrics.binary_structs[Label.LV]
            lv_props = measure.regionprops(measure.label(lv_binary_struct, connectivity=2))
            lv_pixel_count = 0

            if lv_props:  # If the lv is present in the image
                lv_pixel_count = lv_props[0].area

            # Subtract the number of pixels making up the lv segmented area from the holes
            return holes_on_myo_count - lv_pixel_count

        else:  # If myo has no holes, it is open and other metrics should indicate an anatomical error
            return 0


class RightVentricleMetrics(Anatomical2DStructureMetrics):
    """ Class to compute metrics on the segmentation of the right ventricle.
    """

    def __init__(self, segmentation_metrics: Segmentation2DMetrics):
        """
        Args:
            segmentation_metrics: Segmentation2DMetrics, an instance, based on the segmentation for which to compute
                                  anatomical metrics, of the class implementing various segmentation metrics.
        """
        super().__init__(segmentation_metrics, Label.RV)


if __name__ == "__main__":
    database = "pepr"
    dl = PEPRDataModule(data_dir=f'C:/Users/goujat/Documents/thesis/RL4Seg/data/{database}_aphm_cine_train',
                        csv_file=database,
                        gt_column="Gt_0",
                        # subset_frac=0.1,
                        splits_column="split_0",
                        supervised=True,
                        test_frac=0.1)

    dl.setup()

    invalid_rows = []  # liste de tuples (id, slice) à supprimer

    for batch_idx, batch in enumerate(dl.test_dataloader()):
        print(batch['img'].shape)
        print(batch['gt'].shape)
        print(batch['approx_gt'].shape)

        for i in range(batch['gt'].shape[0]):
            print()
            print(batch['slice'][i])
            print(batch['id'][i])
            print(batch['apex_slice'][i])
            print(batch['base_slice'][i])

            is_base = False
            if abs(batch['slice'][i] - batch['base_slice'][i]) <= 2:
                is_base = True
                print("is_base : ", is_base)

            valid, metrics = check_segmentation_validity(np.asarray(batch['gt'][i]),
                                                         np.asarray(batch['vox'][i]),
                                                         is_base)
            print(valid)
            if not valid:
                invalid_rows.append((batch['id'][i], batch['slice'][i]))

                corrected, correction_applied = correct_holes_in_segmentation(np.asarray(batch['gt'][i]), metrics,
                                                                              background_label=Label.BG)
                if correction_applied:
                    patient_id_str = str(batch['id'][i])  # à adapter si c'est déjà une string
                    slice_idx_int = int(batch['slice'][i])  # conversion tensor -> int

                    gt_corrected_dir = f'C:/Users/goujat/Documents/thesis/RL4Seg/data/{database}_aphm_cine_train/gt/'

                    gt_path = gt_corrected_dir + patient_id_str + ".nii.gz"
                    gt_nifti = nib.load(gt_path)
                    gt_data = gt_nifti.get_fdata()

                    gt_data_corrected = gt_data.copy()
                    gt_data_corrected[..., slice_idx_int] = corrected

                    corrected_nifti = nib.Nifti1Image(
                        gt_data_corrected.astype(gt_data.dtype),
                        affine=gt_nifti.affine,
                        header=gt_nifti.header,
                    )

                    nib.save(corrected_nifti, gt_path)

                    print(f"  -> GT corrigée sauvegardée : {gt_path} (slice {slice_idx_int})")

    # --- Mise à jour des CSV ---
    train_csv_path = f'C:/Users/goujat/Documents/thesis/RL4Seg/data/{database}_aphm_cine_train/{database}_slice.csv'
    delete_csv_path = f'C:/Users/goujat/Documents/thesis/RL4Seg/data/{database}_aphm_cine_train/{database}_delete.csv'

    df_train = pd.read_csv(train_csv_path)

    # Construire un DataFrame des lignes invalides
    invalid_ids = [str(x) for x, _ in invalid_rows]
    invalid_slices = [int(s) for _, s in invalid_rows]
    df_invalid_keys = pd.DataFrame({'id': invalid_ids, 'slice': invalid_slices})

    # Masque : lignes de df_train dont (id, slice) est dans invalid_rows
    mask = df_train.set_index(['id', 'slice']).index.isin(
        df_invalid_keys.set_index(['id', 'slice']).index
    )

    df_to_delete = df_train[mask]
    df_train_clean = df_train[~mask]

    # Sauvegarder
    df_train_clean.to_csv(train_csv_path, index=False)

    # Ajouter au fichier de suppression (append si le fichier existe déjà)
    if os.path.exists(delete_csv_path):
        df_delete_existing = pd.read_csv(delete_csv_path)
        df_delete_final = pd.concat([df_delete_existing, df_to_delete], ignore_index=True)
    else:
        df_delete_final = df_to_delete

    df_delete_final.to_csv(delete_csv_path, index=False)

    print(f"{len(df_to_delete)} lignes déplacées vers {database}_delete.csv")
