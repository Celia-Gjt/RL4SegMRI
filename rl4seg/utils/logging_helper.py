import numpy as np
import cv2
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def put_text_to_image(img, text):
    return cv2.putText(img.squeeze(0), "{:.3f}".format(text), (0, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (125), 2)


def log_image(logger, img, title, number=0, img_text=None):
    if logger is None:
        return
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = (img.copy() * (255 / max(img.max(), 1))).astype(np.uint8)

    if img_text is not None:
        img = put_text_to_image(img, img_text)

    logger.experiment.log_image(
        image_data=img,
        name=title,
        step=number
    )


def log_case(logger, img, title, number=0, img_text=None):
    fig, axs = plt.subplots(1, len(img), figsize=(3.2 * len(img), 3.4))
    for ax, im in zip(axs, img):
        if isinstance(im, torch.Tensor):
            im = im.detach().cpu().numpy()
        im = (im.copy() * (255 / max(im.max(), 1))).astype(np.uint8)
        ax.imshow(im[0], cmap='gray')
        ax.axis('off')
    logger.experiment.log_figure(figure=fig, figure_name=title, step=number)
    plt.close(fig)
