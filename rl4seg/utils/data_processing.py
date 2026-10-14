from typing import List, Optional, Tuple, Union

import numpy as np
from skimage.transform import resize


def calculate_new_shape(
        spacing: List[float],
        shape: Union[List[int], Tuple[int]],
        target_spacing: List[float],
) -> List[int]:
    """Calculate the new shape after resampling.

    Args:
        spacing: Original image spacing.
        shape: Original image shape.
        target_spacing: Target image spacing for resampling.

    Returns:
        New shape after resampling.
    """
    spacing_ratio = np.array(spacing) / np.array(target_spacing)

    try:
        new_shape = (spacing_ratio * np.array(shape)).astype(int)
        # Handle the case of pseudo 2D where image's Z spacing is smaller than target Z spacing
        new_shape[new_shape == 0] = 1
    except Exception as e:
        print("Error occurred while calculating new_shape:", e)
        print("Spacing ratio:", spacing_ratio)
        print("Shape:", shape)
    return new_shape.tolist()


def resample_image(
        image: np.ndarray,
        new_shape: Union[List, Tuple],
        anisotropy_flag: bool,
        lowres_axis: Optional[np.ndarray] = None,
        interp_order: int = 3,
        order_z: int = 0,
        verbose: bool = False,
) -> np.ndarray:
    """Resample an image.

    Args:
        image: Image numpy array to be resampled.
        new_shape: Shape after resampling.
        anisotropy_flag: Whether the image is anisotropic.
        lowres_axis: Axis of lowest resolution.
        interp_order: Interpolation order of skimage.transform.resize.
        order_z: Interpolation order for the lowest resolution axis in case of anisotropic image.
        verbose: Whether to log the resampling message.

    Returns:
        Resampled image.
    """
    dtype_data = image.dtype
    shape = np.array(image[0].shape)
    if not np.all(shape == np.array(new_shape)):
        image = image.astype(float)
        resized_channels = []
        if anisotropy_flag:
            if verbose:
                print("Anisotropic image, using separate z resampling")
            axis = lowres_axis[0]
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]
            for image_c in image:
                resized_slices = []
                for i in range(shape[axis]):
                    if axis == 0:
                        image_c_2d_slice = image_c[i]
                    elif axis == 1:
                        image_c_2d_slice = image_c[:, i]
                    else:
                        image_c_2d_slice = image_c[:, :, i]
                    image_c_2d_slice = resize(
                        image_c_2d_slice,
                        new_shape_2d,
                        order=interp_order,
                        mode="edge",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    resized_slices.append(image_c_2d_slice.astype(dtype_data))
                resized = np.stack(resized_slices, axis=axis)
                if not shape[axis] == new_shape[axis]:
                    resized = resize(
                        resized,
                        new_shape,
                        order=order_z,
                        mode="constant",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                resized_channels.append(resized.astype(dtype_data))
        else:
            if verbose:
                print("Not using separate z resampling")
            for image_c in image:
                resized = resize(
                    image_c,
                    new_shape,
                    order=interp_order,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                resized_channels.append(resized.astype(dtype_data))
        reshaped = np.stack(resized_channels, axis=0)
        return reshaped.astype(dtype_data)
    else:
        if verbose:
            print("No resampling necessary")
        return image


def resample_label(
        label: np.ndarray,
        new_shape: Union[List, Tuple],
        anisotropy_flag: bool,
        lowres_axis: Optional[np.ndarray] = None,
        interp_order: int = 1,
        order_z: int = 0,
        verbose: bool = False,
) -> np.ndarray:
    """Resample a label.

    Args:
        label: Label numpy array to be resampled.
        new_shape: Shape after resampling.
        anisotropy_flag: Whether the label is anisotropic.
        lowres_axis: Axis of lowest resolution.
        interp_order: Interpolation order of skimage.transform.resize.
        order_z: Interpolation order for the lowest resolution axis in case of anisotropic label.
        verbose: Whether to log the resampling message.

    Returns:
        Resampled label.
    """
    shape = np.array(label[0].shape)
    if not np.all(shape == np.array(new_shape)):
        reshaped = np.zeros(new_shape, dtype=np.uint8)
        n_class = np.max(label)
        if anisotropy_flag:
            if verbose:
                print("Anisotropic label, using separate z resampling")
            axis = lowres_axis[0]
            depth = shape[axis]
            if axis == 0:
                new_shape_2d = new_shape[1:]
                reshaped_2d = np.zeros((depth, *new_shape_2d), dtype=np.uint8)
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
                reshaped_2d = np.zeros((new_shape_2d[0], depth, new_shape_2d[1]), dtype=np.uint8)
            else:
                new_shape_2d = new_shape[:-1]
                reshaped_2d = np.zeros((*new_shape_2d, depth), dtype=np.uint8)

            for class_ in range(1, int(n_class) + 1):
                for depth_ in range(depth):
                    if axis == 0:
                        mask = label[0, depth_] == class_
                    elif axis == 1:
                        mask = label[0, :, depth_] == class_
                    else:
                        mask = label[0, :, :, depth_] == class_
                    resized_2d = resize(
                        mask.astype(float),
                        new_shape_2d,
                        order=interp_order,
                        mode="edge",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    if axis == 0:
                        reshaped_2d[depth_][resized_2d >= 0.5] = class_
                    elif axis == 1:
                        reshaped_2d[:, depth_][resized_2d >= 0.5] = class_
                    else:
                        reshaped_2d[:, :, depth_][resized_2d >= 0.5] = class_

            if not shape[axis] == new_shape[axis]:
                for class_ in range(1, int(n_class) + 1):
                    mask = reshaped_2d == class_
                    resized = resize(
                        mask.astype(float),
                        new_shape,
                        order=order_z,
                        mode="constant",
                        cval=0,
                        clip=True,
                        anti_aliasing=False,
                    )
                    reshaped[resized >= 0.5] = class_
            else:
                reshaped = reshaped_2d.astype(np.uint8)
        else:
            if verbose:
                print("Not using separate z resampling")
            for class_ in range(1, int(n_class) + 1):
                mask = label[0] == class_
                resized = resize(
                    mask.astype(float),
                    new_shape,
                    order=interp_order,
                    mode="edge",
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped[resized >= 0.5] = class_

        reshaped = np.expand_dims(reshaped, 0)
        return reshaped
    else:
        if verbose:
            print("No resampling necessary")
        return label
