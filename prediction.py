import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

MODEL_PATH = "./../models/age_gender_model.keras"
SIZE = (224, 224)

class Prediction:
    def __init__(self):
        self.model = load_model(MODEL_PATH)

    def preprocess(self, image):
        image = cv2.resize(image, SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        return image

    def predict(self, image):
        pass