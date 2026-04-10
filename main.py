import cv2
import os
import datetime
from app.camera import Camera
from app.face_detector import FaceDetector
from app.prediction import Prediction

def save_image(face):
    directory_name = "data/faces"

    try:
        os.makedirs(directory_name, exist_ok=True)
    except Exception as e:
        print(f"An error occurred while saving image: {e}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"data/faces/face_{timestamp}.jpg"
    cv2.imwrite(filename, face)
    print(f"Saved face snapshot to: {filename}")
        

def main():
    camera = Camera()
    detector = FaceDetector()
    prediction = Prediction()

    try:
        camera.start_camera()

        while True:
            frame = camera.read_frame()
            faces = detector.detect_face(frame)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            largest_face = detector.select_largest_face(faces, frame)
            
            cv2.imshow("Face Detection", frame)

            key = cv2.waitKey(1) & 0xFF   

            if  key == ord('q'):
                break  
            elif key == ord('s'):
                if largest_face is not None:
                    save_image(largest_face)
                else:
                    print("No face detected")
            elif key == ord('p') and largest_face is not None:
                print(prediction.predict(largest_face))
    finally:
        camera.close_camera()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()