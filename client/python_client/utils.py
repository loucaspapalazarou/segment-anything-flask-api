import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import cv2


# Show all the segments and bounding boxes
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        # bbox = ann["bbox"]
        # x, y, width, height = bbox
        # rect = plt.Rectangle(
        #     (x, y), width, height, fill=False, edgecolor="red", linewidth=2
        # )
        # ax.add_patch(rect)
    ax.imshow(img)


# Show single segment
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# Show a start on the selected points
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def parse_image(image):
    if not os.path.exists("tmp"):
        os.makedirs("tmp")
    unique_filename = str(uuid.uuid4()) + ".jpg"
    save_path = os.path.join("tmp", unique_filename)
    image.save(save_path)
    image = cv2.imread(save_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    os.remove(save_path)
    return image
