from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import h5py
import io
import tensorflow.keras.models as models
from PIL import Image
import requests
from tensorflow.keras.preprocessing.image import img_to_array

# Create the Flask app
app = Flask(__name__)
class_names = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
    26: "background"
}
# Load the trained model
model = load_model(r'ASL_CNN_MODEL1.h5')

# Home page route
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle predictions
@app.route("/predict", methods=["POST"])
# Load and resize the image
def predict():
    img_path = request.files["image"]
    image = Image.open(img_path)
    image = image.resize((200, 200))
    image = np.array(image)
    if image.ndim == 2:
        # Grayscale image
        gray_image = image
        print("Image is grayscale")
    elif image.ndim == 3:
        # RGB or BGR image
        if image.shape[2] == 3:
                    # RGB or BGR image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    else:
        print("Image has an unsupported number of dimensions")
        return
    gray_image = np.expand_dims(gray_image, axis=0)
    prediction = model.predict(gray_image)
    predicted_class = np.argmax(prediction)


    # Convert the predicted class to the corresponding alphabet label
    if predicted_class == 26:
        predicted_label = 'background'
    else:
        predicted_label = class_names[predicted_class]
    print(predicted_class)
    print("Predicted Label:", predicted_label)
    return predicted_label
if __name__ == "__main__":
    app.run()
