import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model

MODEL_PATH = "./models/age_gender_model.keras"
TEST_IMAGE_PATH = "./data/UTKFace/20_1_4_20161223230110540.jpg.chip.jpg"
SIZE = (224, 224)
AGE_CATEGORY = [
    (0,12),
    (13, 24),
    (25, 39),
    (40, 59),
    (60, 120),
]

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
        image = self.preprocess(image)
        gender_pred, age_pred = self.model.predict(image)
        gender = "Female" if gender_pred[0][0] > 0.5 else "Male"
        age_category = AGE_CATEGORY[np.argmax(age_pred[0])]
        return {
            "gender": gender,
            "gender confidence": gender_pred[0][0],
            "age category": age_category,
            "age confidence": float(np.max(age_pred[0]))
        }
    
# Test prediction on one image
def main():
    prediction = Prediction()
    image = cv2.imread(TEST_IMAGE_PATH)
    print(prediction.predict(image))

if __name__=="__main__":
    main()
