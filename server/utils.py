import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import cv2
import torch
import torchvision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_models = [
    {"checkpoint": "models/sam_vit_h_4b8939.pth", "type": "vit_h"},
    {"checkpoint": "models/sam_vit_l_0b3195.pth", "type": "vit_l"},
    {"checkpoint": "models/sam_vit_b_01ec64.pth", "type": "vit_b"},
]


# Function to read allowed hosts from the file
def read_allowed_hosts():
    with open("allowed_hosts.txt") as f:
        return [line.strip() for line in f]


def test_env():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())


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
        bbox = ann["bbox"]
        x, y, width, height = bbox
        rect = plt.Rectangle(
            (x, y), width, height, fill=False, edgecolor="red", linewidth=2
        )
        ax.add_patch(rect)
    ax.imshow(img)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


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


def select_model(index):
    sam_checkpoint = sam_models[index]["checkpoint"]
    model_type = sam_models[index]["type"]
    device = "cuda:0"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(model=sam)
    predictor = SamPredictor(sam)

    return mask_generator, predictor


def generate_bounding_box(mask):
    nonzero_indices = np.nonzero(mask)
    ymin, ymax = np.min(nonzero_indices[0]), np.max(nonzero_indices[0])
    xmin, xmax = np.min(nonzero_indices[1]), np.max(nonzero_indices[1])
    bbox = [xmin, ymin, xmax, ymax]
    return bbox
