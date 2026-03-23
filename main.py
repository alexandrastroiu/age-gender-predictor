import cv2
from .app.camera import Camera
from .app.face_detector import FaceDetector

def main():
    camera = Camera()
    detector = FaceDetector()

    try:
        camera.start_camera()

        while True:
            frame = camera.read_frame()
            faces = detector.detect_face(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cv2.imshow("Face Detection", frame)    

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == main:
    main()