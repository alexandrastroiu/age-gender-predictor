import cv2
import numpy as np

SIZE = (224, 224)


# Image preprocessing before prediction
def preprocess_face(face):
    # Resize the image
    face = cv2.resize(face, SIZE)
    # Convert the image to RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # Normalization
    face = face.astype("float32") / 255.0
    # Add an extra dimension to represent batch size
    face = np.expand_dims(face, axis=0)

    return face
