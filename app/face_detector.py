import cv2


class FaceDetector:
    def __init__(self):
        # Incarca un model de detectie faciala antrenat, foloseste clasificatorul Haar Cascade
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect_face(self, frame):
        # Face conversia la grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detecteaza fetele in imagine
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return faces

    def select_largest_face(self, faces, frame):
        if len(faces) == 0:
            return None

        # Returneaza cea mai mare fata detectata in cadru
        col, row, width, height = max(faces, key=lambda face: face[2] * face[3])
        largest_face = frame[row : row + height, col : col + width]
        return largest_face
