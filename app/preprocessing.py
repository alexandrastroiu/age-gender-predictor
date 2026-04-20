import cv2
import numpy as np

SIZE = (224, 224)


# Preprocesarea imaginii inainte de predictie
def preprocess_face(face):
    # Redimensioneaza imaginea
    face = cv2.resize(face, SIZE)
    # Face conversia imaginii de la BGR la RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    # Normalizare
    face = face.astype("float32") / 255.0
    # Adauga o dimensiune suplimentara pentru a reprezenta batch size
    face = np.expand_dims(face, axis=0)

    return face
