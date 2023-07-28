import numpy as np
from flask import Flask, jsonify, request
import time
from utils import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=read_allowed_hosts())


global mask_generator, predictor, selected_model_index
selected_model_index = 1
mask_generator, predictor = select_model(selected_model_index)


@app.route("/", methods=["GET"])
def root():
    return jsonify("Hello from the SAM Flask API!"), 200


@app.route("/available-models", methods=["GET"])
def get_all_models():
    return jsonify(sam_models), 200


@app.route("/current-model", methods=["GET"])
def get_model():
    return jsonify(sam_models[selected_model_index]), 200


@app.route("/change-model", methods=["POST"])
def change_model():
    model_index = request.form.get("index")
    if model_index is None:
        return jsonify("No model index provided"), 400
    if not model_index.isnumeric():
        return jsonify("Index should be an integer"), 400
    model_index = int(model_index)
    if model_index < 0 or model_index > len(sam_models) - 1:
        return jsonify(f"Invalid index. Valid range: 0-{len(sam_models)-1}"), 400
    global mask_generator, predictor, selected_model_index
    selected_model_index = model_index
    mask_generator, predictor = select_model(selected_model_index)
    return jsonify("Success"), 200


# returns a list of bounding boxes in xywh format
@app.route("/mask", methods=["GET"])
def sam_mask():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image = parse_image(image)

    start = time.time()
    masks = mask_generator.generate(image)
    bboxes = []
    for m in masks:
        bboxes.append(m["bbox"])
    print(f"Completed in {time.time()-start}s")

    return bboxes, 200


@app.route("/predict", methods=["GET", "POST"])
def sam_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    x = request.args.get("x")
    y = request.args.get("y")
    if x is None or y is None:
        return jsonify("Both x and y are needed"), 400
    if not x.isnumeric() or not y.isnumeric():
        return jsonify("X and Y should be positive integers"), 400

    image = request.files["image"]
    image = parse_image(image)
    predictor.set_image(image)

    input_point = np.array([[int(x), int(y)]])
    input_label = np.array([1])  # foreground point

    start = time.time()
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    print(f"Completed in {time.time()-start}s")

    highest_score = max(scores)  # Find the highest score
    for _, (mask, score) in enumerate(zip(masks, scores)):
        if score == highest_score:  # Check if the current score is the highest score
            best_mask = mask

    bounding_box = generate_bounding_box(best_mask)
    response = {
        "minX": str(bounding_box[0]),
        "minY": str(bounding_box[1]),
        "maxX": str(bounding_box[2]),
        "maxY": str(bounding_box[3]),
    }

    return response, 200


if __name__ == "__main__":
    app.run(debug=True)
