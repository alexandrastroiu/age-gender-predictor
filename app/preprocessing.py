import cv2
import numpy as np

SIZE = (224, 224)

def preprocess_face(face):
    face = cv2.resize(face, SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=0)

    return face