import requests
from utils import *
import cv2
import json
import numpy as np


def change_model(index: int) -> str:
    url = "http://localhost:5000/change-model"
    data = {"index": index}
    response = requests.post(url, data=data)
    status_code = response.status_code
    response_content = response.content.decode()
    if status_code != 200:
        print(response_content)
        return None
    return response_content


def get_prediction(image: str, x: int, y: int) -> np.array:
    url = "http://localhost:5000/predict"
    params = {"x": x, "y": y}
    files = {"image": open(image, "rb")}
    response = requests.get(url, params=params, files=files)
    status_code = response.status_code
    response_content = response.content.decode()
    if status_code != 200:
        print(response_content)
        return None
    return json.loads(response_content)


def get_masks(image: str) -> list[dict]:
    url = "http://localhost:5000/mask"
    files = {"image": open(image, "rb")}
    response = requests.get(url, files=files)
    status_code = response.status_code
    response_content = response.content.decode()
    if status_code != 200:
        print(response_content)
        return None
    return json.loads(response_content)


# Example usage
if __name__ == "__main__":
    image_path = "images/two_parts_cable.jpg"
    x = 2500
    y = 1290
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    mask = get_prediction(image_path, x, y)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bbox = [int(mask["minX"]), int(mask["minY"]), int(mask["maxX"]), int(mask["maxY"])]

    plt.figure()
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    show_box(bbox, plt.gca())
    plt.axis("off")
    plt.show()
