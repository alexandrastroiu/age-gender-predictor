import cv2

class Camera:
    def __init__(self, webcam_index = 0):
        self.index = webcam_index
        self.capture = None

    # Methods for handling camera operations
    
    def start_camera(self):
        self.capture = cv2.VideoCapture(self.index)

        if not self.capture.isOpened():
            raise RuntimeError("Error: Webcam cannot be opened")
    
    def read_frame(self):
        if self.capture is None:
            raise RuntimeError("Webcam not started")

        ret, frame = self.capture.read()

        if not ret:
            raise RuntimeError("Error: Could not read frame from webcam")
        
        return frame

    def close_camera(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None