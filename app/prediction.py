import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from app.preprocessing import preprocess_face

MODEL_PATH = "./models/age_gender_model.keras"
SIZE = (224, 224)
AGE_CATEGORY = [
    "0 - 12",
    "13 - 24",
    "25 - 39",
    "40 - 59",
    "60 - 120",
]

class Prediction:
    def __init__(self):
        self.model = load_model(MODEL_PATH)

    def predict(self, image):
        image = preprocess_face(image)
        gender_pred, age_pred = self.model.predict(image)
        gender = "Female" if gender_pred[0][0] > 0.5 else "Male"
        gender_confidence = gender_pred[0][0] if gender == "Female" else 1 - gender_pred[0][0]
        age_category = AGE_CATEGORY[np.argmax(age_pred[0])]
        return {
            "Gender": gender,
            "Gender Confidence": gender_confidence,
            "Age Category": age_category,
            "Age Confidence": float(np.max(age_pred[0]))
        }
