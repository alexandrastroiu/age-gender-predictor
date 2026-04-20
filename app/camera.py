import cv2

# Vom folosi biblioteca OpenCV pentru capturarea si procesarea imaginilor de la camera

class Camera:
    def __init__(self, webcam_index=0):
        self.index = webcam_index
        self.capture = None

    # Metode pentru a gestiona operatiile pentru camera

    def start_camera(self):
        # Deschide camera default
        self.capture = cv2.VideoCapture(self.index)

        # Verficare
        if not self.capture.isOpened():
            raise RuntimeError("Error: Webcam cannot be opened")

    def read_frame(self):
        if self.capture is None:
            raise RuntimeError("Webcam not started")

        # Citeste cadru cu cadru de la camera
        # Returneaza statusul operatiei si imaginea
        ret, frame = self.capture.read()

        # Verificare
        if not ret:
            raise RuntimeError("Error: Could not read frame from webcam")

        return frame

    def close_camera(self):
        if self.capture is not None:
            # Elibereaza resursele
            self.capture.release()
            self.capture = None
